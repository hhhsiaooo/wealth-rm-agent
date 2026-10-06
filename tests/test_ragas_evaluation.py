from __future__ import annotations

import pandas as pd
import pytest

from evaluation.ragas_evaluation import (
    FaithfulnessRecord,
    compute_summary,
    score_customer_faithfulness,
    what_explanation_context,
    what_reason_context,
    why_context,
)
from insight.insight_generator import CustomerInsight, KeyField
from rag.product_knowledge import PRODUCTS
from rag.schemas import NBALLMOutput, ProductRecommendation
from tools.customer_tool import EXPECTED_FIELDS

FUND = next(p for p in PRODUCTS if p.metadata.product_category == "Fund")


def make_row(customer_id: str, **overrides) -> dict:
    row = {field: 0 for field in EXPECTED_FIELDS}
    row.update({
        "customer_id": customer_id, "risk_level": "High", "has_fund": 0,
        "investment_intent_score": 92.0, "investment_ratio": 0.0,
        "twd_td_balance": 8_000_000, "priority_rank": 1, "priority_score": 90.0,
    })
    row.update(overrides)
    return row


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "customers.csv"
    pd.DataFrame([make_row("C000104")], columns=EXPECTED_FIELDS).to_csv(path, index=False)
    return path


class FakeRetriever:
    def __init__(self, ids=(FUND.metadata.product_id,)):
        self.ids = ids

    def __call__(self, query, category, risk_levels=None):
        return [{"product_id": i, "document": "", "metadata": {}, "distance": 0.1} for i in self.ids]


class StubLLMClient:
    """Satisfies both LLMClient.__call__ (WHY) and StructuredLLMClient.generate
    (WHAT) -- no network call, deterministic, mirrors StubLLMClient /
    StubStructuredLLM in the existing insight/nba test suites."""

    def __init__(self, insight: CustomerInsight, nba_output: NBALLMOutput):
        self._insight = insight
        self._nba_output = nba_output

    def __call__(self, prompt: str) -> CustomerInsight:
        return self._insight

    def generate(self, prompt, response_model):
        return self._nba_output


def make_insight(customer_id: str = "C000104") -> CustomerInsight:
    return CustomerInsight(
        customer_id=customer_id, priority_rank=1, priority_score=90.0,
        key_fields=[KeyField(field="total_aum", value=8_000_000)],
        insight="客戶資產約800萬元。",
    )


# -- context assembly (pure: no LLM needed) --------------------------------------

def test_why_context_uses_same_rendering_as_prompt():
    ctx = why_context({"customer_id": "C000104", "total_aum": 8000000})
    assert len(ctx) == 1
    assert "8000000" in ctx[0]
    assert "total_aum" in ctx[0]


def test_what_explanation_context_includes_insight_when_present():
    ctx = what_explanation_context(["has_fund=0"], "客戶資產約800萬元。")
    assert ctx == ["has_fund=0", "客戶資產約800萬元。"]


def test_what_explanation_context_omits_insight_when_absent():
    ctx = what_explanation_context(["has_fund=0"], None)
    assert ctx == ["has_fund=0"]


def test_what_reason_context_is_product_data_plus_customer_risk_only():
    ctx = what_reason_context(FUND, "High")
    assert ctx[0] == "客戶風險屬性：High"
    assert len(ctx) == 2
    assert FUND.content.product_name in ctx[1]
    other = next(p for p in PRODUCTS if p.metadata.product_id != FUND.metadata.product_id)
    assert all(other.content.product_name not in c for c in ctx)


# -- score_customer_faithfulness (judge mocked out; real generation is stubbed) --

def test_score_customer_faithfulness_pairs_each_text_with_its_own_context(monkeypatch, csv_path):
    scored = []

    def fake_score_faithfulness(response, contexts, llm, user_input=""):
        scored.append((response, tuple(contexts)))
        return 1.0

    monkeypatch.setattr("evaluation.ragas_evaluation.score_faithfulness", fake_score_faithfulness)

    nba_output = NBALLMOutput(
        explanation="說明文字",
        products=[ProductRecommendation(product_id=FUND.metadata.product_id, product_name="x", reason="理由文字")],
    )
    stub = StubLLMClient(make_insight(), nba_output)
    retriever = FakeRetriever()

    records = score_customer_faithfulness(
        "C000104", stub, judge_llm=None, retriever=retriever, source_path=csv_path,
    )

    stages = [(r.stage, r.text_type) for r in records]
    assert ("WHY", "insight") in stages
    assert ("WHAT", "explanation") in stages
    assert ("WHAT", "reason") in stages
    assert len(scored) == 3  # insight + explanation + 1 reason

    insight_response, insight_ctx = scored[0]
    assert insight_response == "客戶資產約800萬元。"
    assert "total_aum" in insight_ctx[0]  # WHY scored against customer record, not product text

    reason_response, reason_ctx = scored[2]
    assert reason_response == "理由文字"
    assert FUND.content.product_name in reason_ctx[1]  # reason scored against ITS OWN product + customer risk


def test_score_customer_faithfulness_skips_hallucinated_product_id(monkeypatch, csv_path):
    monkeypatch.setattr(
        "evaluation.ragas_evaluation.score_faithfulness",
        lambda response, contexts, llm, user_input="": 1.0,
    )
    nba_output = NBALLMOutput(
        explanation="說明文字",
        products=[ProductRecommendation(product_id="NOT_A_REAL_ID", product_name="x", reason="理由")],
    )
    stub = StubLLMClient(make_insight(), nba_output)
    retriever = FakeRetriever(ids=("NOT_A_REAL_ID",))

    records = score_customer_faithfulness(
        "C000104", stub, judge_llm=None, retriever=retriever, source_path=csv_path,
    )
    assert [r for r in records if r.text_type == "reason"] == []


# -- compute_summary (pure aggregation) -------------------------------------------

def test_compute_summary_groups_by_text_type():
    records = [
        FaithfulnessRecord(stage="WHY", text_type="insight", customer_id="C1", faithfulness=1.0),
        FaithfulnessRecord(stage="WHAT", text_type="explanation", customer_id="C1",
                           recommended_action="A2", faithfulness=0.5),
        FaithfulnessRecord(stage="WHAT", text_type="reason", customer_id="C1",
                           recommended_action="A2", product_id=FUND.metadata.product_id, faithfulness=0.0),
    ]
    summary = compute_summary(records)
    assert summary.total_customers == 1
    assert summary.total_scored_texts == 3
    assert summary.mean_faithfulness_why_insight == 1.0
    assert summary.mean_faithfulness_what_explanation == 0.5
    assert summary.mean_faithfulness_what_reason == 0.0
    assert summary.mean_faithfulness_overall == pytest.approx(0.5)


def test_compute_summary_rejects_empty_list():
    with pytest.raises(ValueError):
        compute_summary([])


def test_compute_summary_excludes_judge_failures_from_means():
    records = [
        FaithfulnessRecord(stage="WHY", text_type="insight", customer_id="C1", faithfulness=1.0),
        FaithfulnessRecord(stage="WHAT", text_type="explanation", customer_id="C1",
                           recommended_action="A2", faithfulness=None),
    ]
    summary = compute_summary(records)
    assert summary.judge_failures == 1
    assert summary.total_scored_texts == 1
    assert summary.mean_faithfulness_overall == 1.0
    assert summary.mean_faithfulness_what_explanation is None


# -- judge retry behavior ------------------------------------------------------------

def test_score_with_retry_recovers_after_one_parse_failure(monkeypatch):
    from langchain_core.exceptions import OutputParserException

    from evaluation.ragas_evaluation import _score_with_retry

    calls = []

    def flaky(response, contexts, llm, user_input=""):
        calls.append(1)
        if len(calls) == 1:
            raise OutputParserException("Invalid json output")
        return 0.75

    monkeypatch.setattr("evaluation.ragas_evaluation.score_faithfulness", flaky)
    assert _score_with_retry("x", ["c"], None, "問題") == 0.75
    assert len(calls) == 2


def test_score_with_retry_returns_none_after_two_parse_failures(monkeypatch):
    from langchain_core.exceptions import OutputParserException

    from evaluation.ragas_evaluation import _score_with_retry

    def always_bad(response, contexts, llm, user_input=""):
        raise OutputParserException("Invalid json output")

    monkeypatch.setattr("evaluation.ragas_evaluation.score_faithfulness", always_bad)
    assert _score_with_retry("x", ["c"], None, "問題") is None


def test_score_customer_faithfulness_stores_judged_text_and_contexts(monkeypatch, csv_path):
    import json

    monkeypatch.setattr(
        "evaluation.ragas_evaluation.score_faithfulness",
        lambda response, contexts, llm, user_input="": 1.0,
    )
    nba_output = NBALLMOutput(
        explanation="說明文字",
        products=[ProductRecommendation(product_id=FUND.metadata.product_id, product_name="x", reason="理由文字")],
    )
    records = score_customer_faithfulness(
        "C000104", StubLLMClient(make_insight(), nba_output), judge_llm=None,
        retriever=FakeRetriever(), source_path=csv_path,
    )
    reason = next(r for r in records if r.text_type == "reason")
    assert reason.response_text == "理由文字"
    assert json.loads(reason.contexts_json)[0] == "客戶風險屬性：High"


def test_rejudge_report_uses_stored_texts_and_contexts(monkeypatch, tmp_path):
    from evaluation.ragas_evaluation import FaithfulnessRecord, _dump_contexts, rejudge_report

    seen = []

    def fake_score(response, contexts, llm, user_input=""):
        seen.append((response, tuple(contexts)))
        return 0.5

    monkeypatch.setattr("evaluation.ragas_evaluation.score_faithfulness", fake_score)

    source = tmp_path / "report.csv"
    row = FaithfulnessRecord(
        stage="WHY", text_type="insight", customer_id="C000104", faithfulness=1.0,
        response_text="客戶資產約800萬元。", contexts_json=_dump_contexts(["total_aum: 8000000"]),
    )
    pd.DataFrame([row.model_dump()]).to_csv(source, index=False, encoding="utf-8-sig")

    records, summary = rejudge_report(source, judge_llm=None)
    assert seen == [("客戶資產約800萬元。", ("total_aum: 8000000",))]
    assert records[0].faithfulness == 0.5
    assert summary.mean_faithfulness_overall == 0.5


def test_rejudge_report_refuses_reports_without_stored_texts(tmp_path):
    from evaluation.ragas_evaluation import rejudge_report

    old = tmp_path / "old.csv"
    pd.DataFrame([{"stage": "WHY", "text_type": "insight", "customer_id": "C1", "faithfulness": 1.0}]).to_csv(
        old, index=False,
    )
    with pytest.raises(ValueError):
        rejudge_report(old, judge_llm=None)


def test_question_for_maps_each_text_type_to_its_own_question():
    from insight.insight_generator import BUSINESS_QUESTION

    from evaluation.ragas_evaluation import question_for

    assert question_for("insight") == BUSINESS_QUESTION
    assert question_for("explanation") != question_for("reason")
    assert question_for("reason") == "為什麼此商品可作為候選？"


def test_score_with_retry_passes_the_question_to_the_judge(monkeypatch):
    from evaluation.ragas_evaluation import _score_with_retry

    seen = []

    def fake(response, contexts, llm, user_input=""):
        seen.append(user_input)
        return 0.9

    monkeypatch.setattr("evaluation.ragas_evaluation.score_faithfulness", fake)
    assert _score_with_retry("x", ["c"], None, "為什麼建議此行動？") == 0.9
    assert seen == ["為什麼建議此行動？"]


def test_score_with_retry_passes_none_through_without_retrying(monkeypatch):
    from evaluation.ragas_evaluation import _score_with_retry

    calls = []

    def no_statements(response, contexts, llm, user_input=""):
        calls.append(1)
        return None

    monkeypatch.setattr("evaluation.ragas_evaluation.score_faithfulness", no_statements)
    assert _score_with_retry("x", ["c"], None, "問題") is None
    assert len(calls) == 1


def test_nan_to_none_turns_nan_into_none_only_for_nan():
    from evaluation.ragas_faithfulness import _nan_to_none

    assert _nan_to_none(float("nan")) is None
    assert _nan_to_none(0.5) == 0.5


def test_score_with_retry_lets_quota_errors_stop_the_run(monkeypatch):
    from evaluation.ragas_evaluation import _score_with_retry

    def quota(response, contexts, llm, user_input=""):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    monkeypatch.setattr("evaluation.ragas_evaluation.score_faithfulness", quota)
    with pytest.raises(RuntimeError):
        _score_with_retry("x", ["c"], None, "問題")
