"""Phase 2 (WHY), Step 8: WHY Evaluation.

Runs the Step7 grounding policy (generate_grounded_customer_insight) across
every customer in outputs/priority_customers_top10.csv and computes the
quantitative metrics agreed for this stage:

- Factual Accuracy (raw): grounded rate on the FIRST LLM attempt only --
  measures raw LLM/prompt quality, before any Step7 correction.
- Factual Accuracy (final): grounded rate after the Step7 policy runs
  ("grounded" + "grounded_after_retry") -- measures what a RM actually sees.
- Hallucination Rate: share of key_fields (first attempt only) whose field
  name does not exist in the source record at all (fabrication).
- Value Mismatch Rate: share of key_fields (first attempt only) whose field
  exists but the value is wrong -- a distinct, less severe failure mode from
  hallucination, tracked separately per the agreed evaluation design.
- data_retrieval_latency_sec / end_to_end_latency_sec: directly measured
  (not estimated by subtraction), per CLAUDE.md's "System Performance"
  section.

This is AI evaluation, not software testing (CLAUDE.md "Evaluation
Principles" keeps the two separate -- pytest covers code correctness on
synthetic fixtures; this script measures real LLM output quality on real
data). main() calls a real LLM once or twice per customer and therefore
costs money -- it is a manual command, never invoked by pytest.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from insight.fact_validator import (  # noqa: E402
    AUDIT_LOG_PATH,
    GroundingOutcome,
    generate_grounded_customer_insight,
)
from insight.insight_generator import GeminiLLMClient, LLMClient, load_env_file, select_llm_client  # noqa: E402
from tools.customer_tool import TOP10_PATH, get_customer_profile  # noqa: E402

OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"
REPORT_PATH = OUTPUTS_DIR / "why_evaluation_report.csv"
SUMMARY_PATH = OUTPUTS_DIR / "why_evaluation_summary.json"

# Failure-mode grouping for the two rates: "unknown_field" is fabrication
# (hallucination); the other three are cases where the field/score exists
# but the LLM's value for it is wrong (value mismatch).
HALLUCINATION_TYPES = {"unknown_field"}
VALUE_MISMATCH_TYPES = {"value_mismatch", "priority_score_mismatch", "priority_rank_mismatch"}


class CustomerEvalRecord(BaseModel):
    customer_id: str
    status: Literal["grounded", "grounded_after_retry", "rejected"]
    attempts: int
    is_grounded_initial: bool
    is_grounded_final: bool
    key_field_count_initial: int
    hallucination_issue_count: int
    value_mismatch_issue_count: int
    data_retrieval_latency_sec: float
    end_to_end_latency_sec: float


class WhyEvaluationSummary(BaseModel):
    total_customers: int
    factual_accuracy_raw: float
    factual_accuracy_final: float
    hallucination_rate: float
    value_mismatch_rate: float
    retry_rate: float
    rejection_rate: float
    mean_data_retrieval_latency_sec: float
    mean_end_to_end_latency_sec: float


def evaluate_customer(
    customer_id: str,
    llm_client: LLMClient,
    source_path: Path = TOP10_PATH,
    audit_log_path: Path = AUDIT_LOG_PATH,
) -> CustomerEvalRecord:
    """Run the Step7 grounding policy for one customer with direct latency
    measurement, and classify the first attempt's issues into the
    hallucination / value-mismatch buckets."""
    t0 = time.perf_counter()
    get_customer_profile(customer_id, source_path=source_path)
    data_retrieval_latency_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    outcome: GroundingOutcome = generate_grounded_customer_insight(
        customer_id, llm_client, source_path=source_path, audit_log_path=audit_log_path
    )
    end_to_end_latency_sec = time.perf_counter() - t1

    initial_issues = outcome.initial_fact_check.issues
    hallucination_count = sum(1 for i in initial_issues if i.type in HALLUCINATION_TYPES)
    value_mismatch_count = sum(1 for i in initial_issues if i.type in VALUE_MISMATCH_TYPES)

    return CustomerEvalRecord(
        customer_id=customer_id,
        status=outcome.status,
        attempts=outcome.attempts,
        is_grounded_initial=outcome.initial_fact_check.is_grounded,
        is_grounded_final=outcome.final_fact_check.is_grounded,
        key_field_count_initial=len(outcome.initial_insight.key_fields),
        hallucination_issue_count=hallucination_count,
        value_mismatch_issue_count=value_mismatch_count,
        data_retrieval_latency_sec=data_retrieval_latency_sec,
        end_to_end_latency_sec=end_to_end_latency_sec,
    )


def compute_summary(records: list[CustomerEvalRecord]) -> WhyEvaluationSummary:
    """Pure aggregation function -- no I/O, no LLM calls -- so it can be
    unit-tested with synthetic records instead of real (paid) evaluation runs."""
    if not records:
        raise ValueError("cannot summarize an empty list of records")

    n = len(records)
    total_key_fields = sum(r.key_field_count_initial for r in records)

    return WhyEvaluationSummary(
        total_customers=n,
        factual_accuracy_raw=sum(r.is_grounded_initial for r in records) / n,
        factual_accuracy_final=sum(r.is_grounded_final for r in records) / n,
        hallucination_rate=(
            sum(r.hallucination_issue_count for r in records) / total_key_fields
            if total_key_fields else 0.0
        ),
        value_mismatch_rate=(
            sum(r.value_mismatch_issue_count for r in records) / total_key_fields
            if total_key_fields else 0.0
        ),
        retry_rate=sum(r.attempts == 2 for r in records) / n,
        rejection_rate=sum(r.status == "rejected" for r in records) / n,
        mean_data_retrieval_latency_sec=sum(r.data_retrieval_latency_sec for r in records) / n,
        mean_end_to_end_latency_sec=sum(r.end_to_end_latency_sec for r in records) / n,
    )


def run_evaluation(
    llm_client: LLMClient,
    source_path: Path = TOP10_PATH,
    audit_log_path: Path = AUDIT_LOG_PATH,
    delay_between_customers_sec: float = 0.0,
) -> tuple[list[CustomerEvalRecord], WhyEvaluationSummary]:
    """Evaluate every customer in source_path. Makes 1-2 real LLM calls per
    customer -- costs money/quota and time; call only from main() or
    explicitly. delay_between_customers_sec throttles requests to stay under
    a provider's free-tier rate limit (e.g. Gemini's free tier is a few
    requests per minute) -- main() sets a conservative default; pass 0 for
    unthrottled providers/tests."""
    df = pd.read_csv(source_path)
    records = []
    for i, customer_id in enumerate(df["customer_id"]):
        if i > 0 and delay_between_customers_sec > 0:
            time.sleep(delay_between_customers_sec)
        print(f"[{i + 1}/{len(df)}] evaluating {customer_id}...")
        records.append(
            evaluate_customer(customer_id, llm_client, source_path=source_path, audit_log_path=audit_log_path)
        )
    return records, compute_summary(records)


def main() -> None:
    load_env_file()
    llm_client = select_llm_client()

    # Gemini's free tier is rate-limited to a handful of requests per minute
    # (observed: 5 RPM for gemini-3.6-flash); throttle to stay under it. No
    # such limit is needed for the paid Anthropic tier.
    delay = 13.0 if isinstance(llm_client, GeminiLLMClient) else 0.0

    records, summary = run_evaluation(llm_client, delay_between_customers_sec=delay)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r.model_dump() for r in records]).to_csv(REPORT_PATH, index=False)
    SUMMARY_PATH.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

    print(f"Evaluated {len(records)} customers -> {REPORT_PATH}")
    print(f"Summary -> {SUMMARY_PATH}")
    print()
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
