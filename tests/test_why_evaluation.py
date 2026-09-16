from __future__ import annotations

import pandas as pd
import pytest

from evaluation.why_evaluation import (
    CustomerEvalRecord,
    compute_summary,
    evaluate_customer,
    run_evaluation,
)
from insight.insight_generator import CustomerInsight, KeyField
from tools.customer_tool import EXPECTED_FIELDS


def make_row(customer_id: str, priority_rank: int = 1, priority_score: float = 90.5, **overrides) -> dict:
    row = {field: 0 for field in EXPECTED_FIELDS}
    row.update({
        "customer_id": customer_id,
        "occupation": "科技業",
        "job_level": "Director",
        "total_aum": 20000000,
        "priority_rank": priority_rank,
        "priority_score": priority_score,
    })
    row.update(overrides)
    return row


def write_csv(tmp_path, rows: list[dict], name="priority_customers_top10.csv"):
    path = tmp_path / name
    pd.DataFrame(rows, columns=EXPECTED_FIELDS).to_csv(path, index=False)
    return path


def make_insight(customer_id="C000104", priority_rank=1, priority_score=90.5, key_fields=None) -> CustomerInsight:
    return CustomerInsight(
        customer_id=customer_id,
        priority_rank=priority_rank,
        priority_score=priority_score,
        key_fields=key_fields or [KeyField(field="total_aum", value=20000000)],
        insight="test insight",
    )


class SequentialLLMClient:
    def __init__(self, responses: list[CustomerInsight]):
        self._responses = list(responses)

    def __call__(self, prompt: str) -> CustomerInsight:
        return self._responses.pop(0)


def make_record(**overrides) -> CustomerEvalRecord:
    defaults = dict(
        customer_id="C000001",
        status="grounded",
        attempts=1,
        is_grounded_initial=True,
        is_grounded_final=True,
        key_field_count_initial=2,
        hallucination_issue_count=0,
        value_mismatch_issue_count=0,
        data_retrieval_latency_sec=0.01,
        end_to_end_latency_sec=0.5,
    )
    defaults.update(overrides)
    return CustomerEvalRecord(**defaults)


# -- compute_summary (pure function, no API calls) ---------------------------

def test_all_grounded_first_try_gives_perfect_scores():
    records = [make_record(customer_id=f"C{i}") for i in range(5)]
    summary = compute_summary(records)

    assert summary.total_customers == 5
    assert summary.factual_accuracy_raw == 1.0
    assert summary.factual_accuracy_final == 1.0
    assert summary.hallucination_rate == 0.0
    assert summary.value_mismatch_rate == 0.0
    assert summary.retry_rate == 0.0
    assert summary.rejection_rate == 0.0


def test_mixed_outcomes_produce_correct_rates():
    records = [
        make_record(customer_id="C1", status="grounded", attempts=1, is_grounded_initial=True, is_grounded_final=True),
        make_record(customer_id="C2", status="grounded_after_retry", attempts=2, is_grounded_initial=False, is_grounded_final=True),
        make_record(customer_id="C3", status="rejected", attempts=2, is_grounded_initial=False, is_grounded_final=False),
        make_record(customer_id="C4", status="grounded", attempts=1, is_grounded_initial=True, is_grounded_final=True),
    ]
    summary = compute_summary(records)

    assert summary.total_customers == 4
    assert summary.factual_accuracy_raw == 0.5       # C1, C4 grounded on attempt 1
    assert summary.factual_accuracy_final == 0.75     # C1, C2, C4 grounded in the end
    assert summary.retry_rate == 0.5                  # C2, C3 had 2 attempts
    assert summary.rejection_rate == 0.25              # C3 only


def test_hallucination_and_value_mismatch_rate_denominator_uses_key_field_counts():
    records = [
        make_record(customer_id="C1", key_field_count_initial=4, hallucination_issue_count=1, value_mismatch_issue_count=1),
        make_record(customer_id="C2", key_field_count_initial=6, hallucination_issue_count=0, value_mismatch_issue_count=2),
    ]
    summary = compute_summary(records)

    total_fields = 4 + 6
    assert summary.hallucination_rate == pytest.approx(1 / total_fields)
    assert summary.value_mismatch_rate == pytest.approx(3 / total_fields)


def test_latency_means_are_averaged_correctly():
    records = [
        make_record(customer_id="C1", data_retrieval_latency_sec=0.10, end_to_end_latency_sec=1.0),
        make_record(customer_id="C2", data_retrieval_latency_sec=0.20, end_to_end_latency_sec=2.0),
    ]
    summary = compute_summary(records)

    assert summary.mean_data_retrieval_latency_sec == pytest.approx(0.15)
    assert summary.mean_end_to_end_latency_sec == pytest.approx(1.5)


def test_compute_summary_rejects_empty_list():
    with pytest.raises(ValueError):
        compute_summary([])


# -- evaluate_customer (stub LLM, no real API calls) --------------------------

def test_evaluate_customer_grounded_first_try(tmp_path):
    path = write_csv(tmp_path, [make_row("C000104", 1, 90.5, total_aum=20000000)])
    audit_path = tmp_path / "audit.jsonl"
    stub = SequentialLLMClient([make_insight()])

    record = evaluate_customer("C000104", stub, source_path=path, audit_log_path=audit_path)

    assert record.status == "grounded"
    assert record.attempts == 1
    assert record.is_grounded_initial is True
    assert record.is_grounded_final is True
    assert record.hallucination_issue_count == 0
    assert record.value_mismatch_issue_count == 0
    assert record.key_field_count_initial == 1
    assert record.data_retrieval_latency_sec >= 0
    assert record.end_to_end_latency_sec >= 0


def test_evaluate_customer_classifies_hallucination_vs_value_mismatch(tmp_path):
    path = write_csv(tmp_path, [make_row("C000104", 1, 90.5, total_aum=20000000)])
    audit_path = tmp_path / "audit.jsonl"
    wrong = make_insight(key_fields=[
        KeyField(field="total_aum", value=1),        # value mismatch (field exists)
        KeyField(field="not_a_real_field", value=1),  # hallucination (field doesn't exist)
    ])
    # Retry also fails, so status ends up "rejected" -- but the initial
    # attempt's classification is what we're checking here.
    stub = SequentialLLMClient([wrong, wrong])

    record = evaluate_customer("C000104", stub, source_path=path, audit_log_path=audit_path)

    assert record.is_grounded_initial is False
    assert record.hallucination_issue_count == 1
    assert record.value_mismatch_issue_count == 1
    assert record.key_field_count_initial == 2


# -- run_evaluation (multiple customers, stub LLM) -----------------------

def test_run_evaluation_covers_every_customer(tmp_path):
    path = write_csv(tmp_path, [
        make_row("C000001", 1, 90.5, total_aum=20000000),
        make_row("C000002", 2, 85.0, total_aum=15000000),
        make_row("C000003", 3, 80.0, total_aum=10000000),
    ])
    audit_path = tmp_path / "audit.jsonl"

    class PerCustomerStub:
        """Returns a correct insight tailored to whichever customer_id is requested."""

        def __init__(self):
            self.df = pd.read_csv(path)

        def __call__(self, prompt: str) -> CustomerInsight:
            for _, row in self.df.iterrows():
                if row["customer_id"] in prompt:
                    return make_insight(
                        customer_id=row["customer_id"],
                        priority_rank=int(row["priority_rank"]),
                        priority_score=float(row["priority_score"]),
                        key_fields=[KeyField(field="total_aum", value=int(row["total_aum"]))],
                    )
            raise AssertionError("no matching customer_id found in prompt")

    records, summary = run_evaluation(PerCustomerStub(), source_path=path, audit_log_path=audit_path)

    assert len(records) == 3
    assert {r.customer_id for r in records} == {"C000001", "C000002", "C000003"}
    assert summary.total_customers == 3
    assert summary.factual_accuracy_final == 1.0
