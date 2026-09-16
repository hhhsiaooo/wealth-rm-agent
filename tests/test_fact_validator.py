from __future__ import annotations

import json

import pandas as pd
import pytest

from insight.fact_validator import (
    FactCheckResult,
    GroundingOutcome,
    generate_grounded_customer_insight,
    validate_customer_insight,
)
from insight.insight_generator import CustomerInsight, InsightGenerationError, KeyField
from tools.customer_tool import EXPECTED_FIELDS, TOP10_PATH, CustomerNotFoundError


def make_row(customer_id: str, priority_rank: int = 1, priority_score: float = 90.5, **overrides) -> dict:
    row = {field: 0 for field in EXPECTED_FIELDS}
    row.update({
        "customer_id": customer_id,
        "occupation": "科技業",
        "job_level": "Director",
        "gender": "Male",
        "age": 37,
        "risk_level": "High",
        "total_aum": 20000000,
        "twd_demand_ratio": 0.6,
        "priority_rank": priority_rank,
        "priority_score": priority_score,
    })
    row.update(overrides)
    return row


def write_csv(tmp_path, rows: list[dict]):
    path = tmp_path / "priority_customers_top10.csv"
    pd.DataFrame(rows, columns=EXPECTED_FIELDS).to_csv(path, index=False)
    return path


def make_insight(customer_id="C000104", priority_rank=1, priority_score=90.5, key_fields=None, insight_text="test") -> CustomerInsight:
    return CustomerInsight(
        customer_id=customer_id,
        priority_rank=priority_rank,
        priority_score=priority_score,
        key_fields=key_fields or [KeyField(field="total_aum", value=20000000)],
        insight=insight_text,
    )


class SequentialLLMClient:
    """Returns responses[0] on the first call, responses[1] on the second, etc."""

    def __init__(self, responses: list[CustomerInsight]):
        self._responses = list(responses)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> CustomerInsight:
        self.prompts.append(prompt)
        return self._responses.pop(0)


# -- fully grounded (validate_customer_insight) -------------------------

def test_fully_correct_insight_is_grounded(tmp_path):
    path = write_csv(tmp_path, [make_row("C000104", 1, 90.5)])
    insight = make_insight()

    result = validate_customer_insight(insight, "C000104", source_path=path)

    assert isinstance(result, FactCheckResult)
    assert result.is_grounded is True
    assert result.issues == []
    assert result.customer_id == "C000104"


def test_multiple_key_fields_all_correct(tmp_path):
    path = write_csv(tmp_path, [make_row("C000104", 1, 90.5, twd_demand_ratio=0.6, occupation="科技業")])
    insight = make_insight(key_fields=[
        KeyField(field="total_aum", value=20000000),
        KeyField(field="twd_demand_ratio", value=0.6),
        KeyField(field="occupation", value="科技業"),
    ])

    result = validate_customer_insight(insight, "C000104", source_path=path)
    assert result.is_grounded is True
    assert result.issues == []


# -- mismatches: now typed GroundingIssue objects -----------------------

def test_unknown_field_name_is_flagged_as_hallucination_type(tmp_path):
    path = write_csv(tmp_path, [make_row("C000104")])
    insight = make_insight(key_fields=[KeyField(field="not_a_real_field", value=123)])

    result = validate_customer_insight(insight, "C000104", source_path=path)

    assert result.is_grounded is False
    assert len(result.issues) == 1
    assert result.issues[0].type == "unknown_field"
    assert "not_a_real_field" in result.issues[0].detail


def test_wrong_value_is_flagged_as_value_mismatch_type(tmp_path):
    path = write_csv(tmp_path, [make_row("C000104", 1, 90.5, total_aum=20000000)])
    insight = make_insight(key_fields=[KeyField(field="total_aum", value=99999999)])

    result = validate_customer_insight(insight, "C000104", source_path=path)

    assert result.is_grounded is False
    assert result.issues[0].type == "value_mismatch"
    assert "total_aum" in result.issues[0].detail


def test_priority_score_mismatch_type(tmp_path):
    path = write_csv(tmp_path, [make_row("C000104", 1, 90.5)])
    insight = make_insight(priority_score=77.0)

    result = validate_customer_insight(insight, "C000104", source_path=path)

    assert result.is_grounded is False
    assert any(i.type == "priority_score_mismatch" for i in result.issues)


def test_priority_rank_mismatch_type(tmp_path):
    path = write_csv(tmp_path, [make_row("C000104", 1, 90.5)])
    insight = make_insight(priority_rank=5)

    result = validate_customer_insight(insight, "C000104", source_path=path)

    assert result.is_grounded is False
    assert any(i.type == "priority_rank_mismatch" for i in result.issues)


def test_customer_id_mismatch_type(tmp_path):
    path = write_csv(tmp_path, [make_row("C000104", 1, 90.5), make_row("C000105", 2, 80.0)])
    insight = make_insight(customer_id="C000105")

    result = validate_customer_insight(insight, "C000104", source_path=path)

    assert result.is_grounded is False
    assert any(i.type == "customer_id_mismatch" for i in result.issues)


def test_multiple_issues_are_all_reported(tmp_path):
    path = write_csv(tmp_path, [make_row("C000104", 1, 90.5, total_aum=20000000)])
    insight = make_insight(
        priority_rank=99,
        priority_score=1.0,
        key_fields=[KeyField(field="total_aum", value=0), KeyField(field="ghost_field", value=1)],
    )

    result = validate_customer_insight(insight, "C000104", source_path=path)

    assert result.is_grounded is False
    assert len(result.issues) == 4
    types = {i.type for i in result.issues}
    assert types == {"value_mismatch", "unknown_field", "priority_score_mismatch", "priority_rank_mismatch"}


# -- numeric tolerance -------------------------------------------------------

def test_float_precision_does_not_cause_false_mismatch(tmp_path):
    path = write_csv(tmp_path, [make_row("C000104", 1, 90.5)])
    insight = make_insight(priority_score=90.50000000001)

    result = validate_customer_insight(insight, "C000104", source_path=path)
    assert result.is_grounded is True


def test_int_vs_float_representation_matches(tmp_path):
    path = write_csv(tmp_path, [make_row("C000104", 1, 90.5, total_aum=20000000)])
    insight = make_insight(key_fields=[KeyField(field="total_aum", value=20000000.0)])

    result = validate_customer_insight(insight, "C000104", source_path=path)
    assert result.is_grounded is True


# -- propagated errors -------------------------------------------------------

def test_unknown_requested_customer_id_propagates(tmp_path):
    path = write_csv(tmp_path, [make_row("C000104")])
    insight = make_insight()

    with pytest.raises(CustomerNotFoundError):
        validate_customer_insight(insight, "C999999", source_path=path)


# -- Step7: generate_grounded_customer_insight (grounding policy) -----------

def test_grounded_on_first_attempt(tmp_path):
    path = write_csv(tmp_path, [make_row("C000104", 1, 90.5, total_aum=20000000)])
    audit_path = tmp_path / "audit.jsonl"
    stub = SequentialLLMClient([make_insight()])

    outcome = generate_grounded_customer_insight("C000104", stub, source_path=path, audit_log_path=audit_path)

    assert isinstance(outcome, GroundingOutcome)
    assert outcome.status == "grounded"
    assert outcome.attempts == 1
    assert outcome.insight is not None
    assert outcome.fallback_message is None
    assert outcome.initial_fact_check.is_grounded is True
    assert outcome.final_fact_check.is_grounded is True
    assert len(stub.prompts) == 1  # no retry needed


def test_grounded_after_corrective_retry(tmp_path):
    path = write_csv(tmp_path, [make_row("C000104", 1, 90.5, total_aum=20000000)])
    audit_path = tmp_path / "audit.jsonl"
    wrong = make_insight(key_fields=[KeyField(field="total_aum", value=1)])
    correct = make_insight(key_fields=[KeyField(field="total_aum", value=20000000)])
    stub = SequentialLLMClient([wrong, correct])

    outcome = generate_grounded_customer_insight("C000104", stub, source_path=path, audit_log_path=audit_path)

    assert outcome.status == "grounded_after_retry"
    assert outcome.attempts == 2
    assert outcome.insight == correct
    assert outcome.fallback_message is None
    assert outcome.initial_fact_check.is_grounded is False
    assert outcome.final_fact_check.is_grounded is True
    assert len(stub.prompts) == 2


def test_correction_prompt_includes_the_specific_issue(tmp_path):
    path = write_csv(tmp_path, [make_row("C000104", 1, 90.5, total_aum=20000000)])
    audit_path = tmp_path / "audit.jsonl"
    wrong = make_insight(key_fields=[KeyField(field="total_aum", value=1)])
    correct = make_insight(key_fields=[KeyField(field="total_aum", value=20000000)])
    stub = SequentialLLMClient([wrong, correct])

    generate_grounded_customer_insight("C000104", stub, source_path=path, audit_log_path=audit_path)

    correction_prompt = stub.prompts[1]
    assert "total_aum" in correction_prompt
    assert "mismatch" in correction_prompt


def test_rejected_after_failed_retry(tmp_path):
    path = write_csv(tmp_path, [make_row("C000104", 1, 90.5, total_aum=20000000)])
    audit_path = tmp_path / "audit.jsonl"
    wrong1 = make_insight(key_fields=[KeyField(field="total_aum", value=1)])
    wrong2 = make_insight(key_fields=[KeyField(field="total_aum", value=2)])
    stub = SequentialLLMClient([wrong1, wrong2])

    outcome = generate_grounded_customer_insight("C000104", stub, source_path=path, audit_log_path=audit_path)

    assert outcome.status == "rejected"
    assert outcome.attempts == 2
    assert outcome.insight is None  # never shown to the user
    assert outcome.fallback_message == "AI 洞察未通過事實查核，請直接查閱客戶原始資料。"
    assert outcome.final_fact_check.is_grounded is False


def test_rejects_wrong_type_on_corrective_retry(tmp_path):
    path = write_csv(tmp_path, [make_row("C000104", 1, 90.5, total_aum=20000000)])
    audit_path = tmp_path / "audit.jsonl"
    wrong = make_insight(key_fields=[KeyField(field="total_aum", value=1)])

    class BadRetryClient:
        def __init__(self):
            self.calls = 0

        def __call__(self, prompt: str):
            self.calls += 1
            if self.calls == 1:
                return wrong
            return {"not": "a CustomerInsight"}

    with pytest.raises(InsightGenerationError):
        generate_grounded_customer_insight("C000104", BadRetryClient(), source_path=path, audit_log_path=audit_path)


# -- audit log --------------------------------------------------------------

def test_audit_log_written_for_grounded_case(tmp_path):
    path = write_csv(tmp_path, [make_row("C000104", 1, 90.5, total_aum=20000000)])
    audit_path = tmp_path / "audit.jsonl"
    stub = SequentialLLMClient([make_insight()])

    generate_grounded_customer_insight("C000104", stub, source_path=path, audit_log_path=audit_path)

    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2  # 1 attempt record + 1 final-status record
    records = [json.loads(line) for line in lines]
    assert records[0]["attempt"] == 1
    assert records[0]["is_grounded"] is True
    assert records[1]["final_status"] == "grounded"


def test_audit_log_written_for_rejected_case_has_both_attempts(tmp_path):
    path = write_csv(tmp_path, [make_row("C000104", 1, 90.5, total_aum=20000000)])
    audit_path = tmp_path / "audit.jsonl"
    wrong1 = make_insight(key_fields=[KeyField(field="total_aum", value=1)])
    wrong2 = make_insight(key_fields=[KeyField(field="total_aum", value=2)])
    stub = SequentialLLMClient([wrong1, wrong2])

    generate_grounded_customer_insight("C000104", stub, source_path=path, audit_log_path=audit_path)

    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(line) for line in lines]
    assert len(records) == 3  # attempt 1, attempt 2, final status
    assert records[0]["attempt"] == 1
    assert records[1]["attempt"] == 2
    assert records[2]["final_status"] == "rejected"


def test_audit_log_appends_across_multiple_customers(tmp_path):
    path = write_csv(tmp_path, [make_row("C000104", 1, 90.5, total_aum=20000000), make_row("C000105", 2, 80.0, total_aum=20000000)])
    audit_path = tmp_path / "audit.jsonl"

    generate_grounded_customer_insight("C000104", SequentialLLMClient([make_insight("C000104", 1, 90.5)]), source_path=path, audit_log_path=audit_path)
    generate_grounded_customer_insight("C000105", SequentialLLMClient([make_insight("C000105", 2, 80.0)]), source_path=path, audit_log_path=audit_path)

    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4  # 2 records per customer


# -- end-to-end on actual WHO output (validate_customer_insight only) -------

@pytest.mark.skipif(not TOP10_PATH.exists(), reason="priority_customers_top10.csv not generated yet")
def test_end_to_end_accurate_insight_is_grounded():
    df = pd.read_csv(TOP10_PATH)
    row = df.loc[df["priority_rank"].idxmin()]

    accurate_insight = make_insight(
        customer_id=row["customer_id"],
        priority_rank=int(row["priority_rank"]),
        priority_score=float(row["priority_score"]),
        key_fields=[KeyField(field="total_aum", value=int(row["total_aum"]))],
    )

    result = validate_customer_insight(accurate_insight, row["customer_id"])
    assert result.is_grounded is True


@pytest.mark.skipif(not TOP10_PATH.exists(), reason="priority_customers_top10.csv not generated yet")
def test_end_to_end_corrupted_insight_is_not_grounded():
    df = pd.read_csv(TOP10_PATH)
    row = df.loc[df["priority_rank"].idxmin()]

    corrupted_insight = make_insight(
        customer_id=row["customer_id"],
        priority_rank=int(row["priority_rank"]),
        priority_score=float(row["priority_score"]),
        key_fields=[KeyField(field="total_aum", value=int(row["total_aum"]) + 1)],
    )

    result = validate_customer_insight(corrupted_insight, row["customer_id"])
    assert result.is_grounded is False
