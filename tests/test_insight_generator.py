from __future__ import annotations

import os

import pandas as pd
import pydantic
import pytest

from insight.insight_generator import (
    CustomerInsight,
    InsightGenerationError,
    KeyField,
    build_prompt,
    call_with_retry,
    generate_customer_insight,
    load_env_file,
    resolve_model,
)
from tools.customer_tool import EXPECTED_FIELDS, TOP10_PATH, CustomerNotFoundError


def make_row(customer_id: str, priority_rank: int = 1, priority_score: float = 90.5) -> dict:
    row = {field: 0 for field in EXPECTED_FIELDS}
    row.update({
        "customer_id": customer_id,
        "occupation": "科技業",
        "job_level": "Director",
        "gender": "Male",
        "age": 37,
        "risk_level": "High",
        "priority_rank": priority_rank,
        "priority_score": priority_score,
    })
    return row


def write_csv(tmp_path, rows: list[dict]):
    path = tmp_path / "priority_customers_top10.csv"
    pd.DataFrame(rows, columns=EXPECTED_FIELDS).to_csv(path, index=False)
    return path


class StubLLMClient:
    """Deterministic fake LLM -- no network call, used for all default tests."""

    def __init__(self, response: CustomerInsight | None = None, capture: bool = False):
        self._response = response
        self.capture = capture
        self.last_prompt: str | None = None

    def __call__(self, prompt: str) -> CustomerInsight:
        if self.capture:
            self.last_prompt = prompt
        if self._response is not None:
            return self._response
        return CustomerInsight(
            customer_id="C000104",
            priority_rank=1,
            priority_score=90.5,
            key_fields=[KeyField(field="total_aum", value=20000000)],
            insight="客戶資產規模高，近期有投資意圖訊號。",
        )


class WrongTypeLLMClient:
    def __call__(self, prompt: str):
        return {"not": "a CustomerInsight"}


# -- Pydantic schema -------------------------------------------------------

def test_customer_insight_requires_at_least_one_key_field():
    with pytest.raises(pydantic.ValidationError):
        CustomerInsight(
            customer_id="C000001", priority_rank=1, priority_score=90.0,
            key_fields=[], insight="test",
        )


def test_customer_insight_valid_instance():
    insight = CustomerInsight(
        customer_id="C000001", priority_rank=1, priority_score=90.0,
        key_fields=[KeyField(field="total_aum", value=8000000)],
        insight="test insight",
    )
    assert insight.customer_id == "C000001"
    assert insight.key_fields[0].field == "total_aum"


def test_key_field_accepts_any_value_type():
    assert KeyField(field="has_fund", value=0).value == 0
    assert KeyField(field="occupation", value="科技業").value == "科技業"
    assert KeyField(field="twd_demand_ratio", value=0.6).value == 0.6


# -- build_prompt -----------------------------------------------------------

def test_prompt_contains_business_question():
    profile = make_row("C000001")
    prompt = build_prompt(profile)
    assert "為什麼這位客戶值得優先聯繫" in prompt


def test_prompt_contains_full_customer_record():
    profile = make_row("C000001")
    prompt = build_prompt(profile)
    for field in EXPECTED_FIELDS:
        assert field in prompt
    assert "C000001" in prompt


def test_prompt_contains_all_must_not_rules():
    profile = make_row("C000001")
    prompt = build_prompt(profile)
    for phrase in ["不可捏造", "不可修改", "不可假設", "不可創造新的業務規則", "不可重新計算", "不可推薦", "不可做出最終適合度判斷"]:
        assert phrase in prompt


def test_prompt_only_contains_one_customer():
    profile = make_row("C000001")
    prompt = build_prompt(profile)
    assert prompt.count("C000001") >= 1
    assert "C000002" not in prompt


# -- generate_customer_insight -----------------------------------------------

def test_generate_customer_insight_returns_llm_response(tmp_path):
    path = write_csv(tmp_path, [make_row("C000104", 1, 90.5)])
    stub = StubLLMClient()
    result = generate_customer_insight("C000104", stub, source_path=path)
    assert isinstance(result, CustomerInsight)
    assert result.customer_id == "C000104"


def test_generate_customer_insight_passes_full_profile_to_llm(tmp_path):
    path = write_csv(tmp_path, [make_row("C000104", 1, 90.5)])
    stub = StubLLMClient(capture=True)
    generate_customer_insight("C000104", stub, source_path=path)
    assert stub.last_prompt is not None
    assert "total_aum" in stub.last_prompt
    assert "priority_score" in stub.last_prompt


def test_generate_customer_insight_unknown_customer_id_propagates(tmp_path):
    path = write_csv(tmp_path, [make_row("C000104")])
    stub = StubLLMClient()
    with pytest.raises(CustomerNotFoundError):
        generate_customer_insight("C999999", stub, source_path=path)


def test_generate_customer_insight_rejects_wrong_return_type(tmp_path):
    path = write_csv(tmp_path, [make_row("C000104")])
    with pytest.raises(InsightGenerationError):
        generate_customer_insight("C000104", WrongTypeLLMClient(), source_path=path)


def test_generate_customer_insight_does_not_send_entire_csv(tmp_path):
    path = write_csv(tmp_path, [make_row("C000001", 1), make_row("C000002", 2)])
    stub = StubLLMClient(capture=True)
    generate_customer_insight("C000001", stub, source_path=path)
    assert "C000002" not in stub.last_prompt


# -- load_env_file ------------------------------------------------------

def test_load_env_file_sets_new_var(tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_INSIGHT_VAR", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("TEST_INSIGHT_VAR=hello\n# comment\n\nOTHER=1\n", encoding="utf-8")

    load_env_file(env_path)

    assert os.environ["TEST_INSIGHT_VAR"] == "hello"
    del os.environ["TEST_INSIGHT_VAR"]
    del os.environ["OTHER"]


def test_load_env_file_does_not_override_existing_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_INSIGHT_VAR", "already_set")
    env_path = tmp_path / ".env"
    env_path.write_text("TEST_INSIGHT_VAR=from_file\n", encoding="utf-8")

    load_env_file(env_path)

    assert os.environ["TEST_INSIGHT_VAR"] == "already_set"


def test_load_env_file_missing_file_is_a_noop(tmp_path):
    load_env_file(tmp_path / "does_not_exist.env")  # must not raise


# -- end-to-end on actual WHO output (stub LLM, no real API call) -----------

@pytest.mark.skipif(not TOP10_PATH.exists(), reason="priority_customers_top10.csv not generated yet")
def test_end_to_end_on_actual_top10_file_with_stub():
    df = pd.read_csv(TOP10_PATH)
    top_row = df.loc[df["priority_rank"].idxmin()]

    stub = StubLLMClient(response=CustomerInsight(
        customer_id=top_row["customer_id"],
        priority_rank=int(top_row["priority_rank"]),
        priority_score=float(top_row["priority_score"]),
        key_fields=[KeyField(field="total_aum", value=int(top_row["total_aum"]))],
        insight="stub insight for integration test",
    ))

    result = generate_customer_insight(top_row["customer_id"], stub)
    assert result.customer_id == top_row["customer_id"]
    assert result.priority_rank == int(top_row["priority_rank"])


# -- real API integration test (opt-in only) ---------------------------------

@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set -- skipping real Anthropic API call",
)
@pytest.mark.skipif(not TOP10_PATH.exists(), reason="priority_customers_top10.csv not generated yet")
def test_real_anthropic_api_call_returns_valid_insight():
    from insight.insight_generator import AnthropicLLMClient

    df = pd.read_csv(TOP10_PATH)
    customer_id = df.loc[df["priority_rank"].idxmin(), "customer_id"]

    result = generate_customer_insight(customer_id, AnthropicLLMClient())

    assert isinstance(result, CustomerInsight)
    assert result.customer_id == customer_id
    assert len(result.key_fields) >= 1
    assert len(result.insight) > 0


@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set -- skipping real Gemini API call",
)
@pytest.mark.skipif(not TOP10_PATH.exists(), reason="priority_customers_top10.csv not generated yet")
def test_real_gemini_api_call_returns_valid_insight():
    from insight.insight_generator import GeminiLLMClient

    df = pd.read_csv(TOP10_PATH)
    customer_id = df.loc[df["priority_rank"].idxmin(), "customer_id"]

    result = generate_customer_insight(customer_id, GeminiLLMClient())

    assert isinstance(result, CustomerInsight)
    assert result.customer_id == customer_id
    assert len(result.key_fields) >= 1
    assert len(result.insight) > 0


# -- call_with_retry (transient provider errors) --------------------------------

class FakeApiError(Exception):
    def __init__(self, code):
        super().__init__(f"api error {code}")
        self.code = code


def make_flaky(fail_codes, result="ok"):
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] <= len(fail_codes):
            raise FakeApiError(fail_codes[calls["n"] - 1])
        return result

    return fn, calls


def test_call_with_retry_recovers_from_transient_errors():
    fn, calls = make_flaky([503, 429])
    sleeps = []
    assert call_with_retry(fn, delays=(1.0, 2.0, 3.0), sleep=sleeps.append) == "ok"
    assert calls["n"] == 3 and sleeps == [1.0, 2.0]


def test_call_with_retry_gives_up_after_the_last_delay():
    fn, calls = make_flaky([503, 503, 503, 503, 503])
    sleeps = []
    with pytest.raises(FakeApiError):
        call_with_retry(fn, delays=(1.0, 2.0), sleep=sleeps.append)
    assert calls["n"] == 3 and sleeps == [1.0, 2.0]


def test_call_with_retry_does_not_retry_non_transient_errors():
    fn, calls = make_flaky([400])
    with pytest.raises(FakeApiError):
        call_with_retry(fn, delays=(1.0,), sleep=lambda s: None)
    assert calls["n"] == 1


def test_call_with_retry_does_not_wait_on_a_per_day_quota():
    class DailyQuota(FakeApiError):
        def __str__(self):
            return "429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel-FreeTier"

    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise DailyQuota(429)

    sleeps = []
    with pytest.raises(FakeApiError):
        call_with_retry(fn, delays=(1.0, 2.0), sleep=sleeps.append)
    assert calls["n"] == 1 and sleeps == []


def test_resolve_model_precedence(monkeypatch):
    monkeypatch.delenv("SOME_MODEL", raising=False)
    assert resolve_model(None, "SOME_MODEL", "default-model") == "default-model"
    monkeypatch.setenv("SOME_MODEL", "env-model")
    assert resolve_model(None, "SOME_MODEL", "default-model") == "env-model"
    assert resolve_model("explicit-model", "SOME_MODEL", "default-model") == "explicit-model"
