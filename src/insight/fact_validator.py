"""Phase 2 (WHY), Step 6+7: Fact Validation and Grounding Policy.

Step 6 -- validate_customer_insight(): checks every value an LLM-produced
CustomerInsight claims against the original 43-field customer record
(retrieved via tools.customer_tool.get_customer_profile) -- the same source
insight_generator.py used to build the prompt.

Checks performed (per docs/ai_product_design.md #### Fact Validation):
- customer_id exists and matches the requested customer
- every key_fields[].field exists in the source record
- every key_fields[].value matches the source value exactly
- priority_score matches the source value
- priority_rank matches the source value

Step 7 -- generate_grounded_customer_insight(): the agreed grounding policy,
generate -> validate -> (if ungrounded) corrective retry with the specific
issues fed back -> validate again -> (if still ungrounded) reject and return
a fallback message instead of the insight. Every attempt is appended to a
JSONL audit log for human review, regardless of outcome.

This module does not modify WHO ranking logic and does not implement
WHAT / RAG / Agent orchestration.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from insight.insight_generator import (  # noqa: E402
    CustomerInsight,
    InsightGenerationError,
    LLMClient,
    build_correction_prompt,
    generate_customer_insight,
    load_env_file,
    select_llm_client,
)
from tools.customer_tool import CustomerDataError, TOP10_PATH, get_customer_profile  # noqa: E402

AUDIT_LOG_PATH = Path(__file__).resolve().parents[2] / "outputs" / "grounding_audit_log.jsonl"

FALLBACK_MESSAGE = "AI 洞察未通過事實查核，請直接查閱客戶原始資料。"

IssueType = Literal[
    "customer_id_mismatch",
    "unknown_field",           # hallucination: field name not in the source record at all
    "value_mismatch",          # field exists, but the LLM's value is wrong
    "priority_score_mismatch",
    "priority_rank_mismatch",
]


class GroundingIssue(BaseModel):
    type: IssueType
    detail: str


class FactCheckResult(BaseModel):
    customer_id: str
    is_grounded: bool
    issues: list[GroundingIssue]


class GroundingOutcome(BaseModel):
    """Result of the Step7 grounding policy for one customer.

    initial_insight/initial_fact_check are always the first attempt's raw
    result (useful for measuring raw LLM quality, e.g. in Step8 evaluation);
    final_fact_check is the result actually used to decide the outcome (same
    as initial_fact_check when no retry was needed). `insight` is the one
    actually considered safe to show the user -- None when status is
    "rejected".
    """

    customer_id: str
    status: Literal["grounded", "grounded_after_retry", "rejected"]
    insight: Optional[CustomerInsight] = None
    initial_insight: CustomerInsight
    initial_fact_check: FactCheckResult
    final_fact_check: FactCheckResult
    attempts: int
    fallback_message: Optional[str] = None


def _values_match(source_value: Any, llm_value: Any) -> bool:
    """Compare a source record value to an LLM-returned value.

    Numeric values are compared with a small tolerance so JSON round-trip
    float formatting (e.g. 90.5 vs 90.50000000001) doesn't produce a false
    mismatch; everything else must match exactly (including type-sensitive
    cases like "0" vs 0).
    """
    if isinstance(source_value, bool) or isinstance(llm_value, bool):
        return source_value == llm_value
    if isinstance(source_value, (int, float)) and isinstance(llm_value, (int, float)):
        return math.isclose(float(source_value), float(llm_value), rel_tol=1e-9, abs_tol=1e-6)
    return source_value == llm_value


def validate_customer_insight(
    insight: CustomerInsight,
    requested_customer_id: str,
    source_path: Path = TOP10_PATH,
) -> FactCheckResult:
    """Check a CustomerInsight against the source 43-field customer record.

    Raises:
        CustomerNotFoundError / CustomerSchemaError / CustomerDataError:
            propagated unchanged from get_customer_profile() -- a missing or
            malformed source record is a data/system error, not a "fact
            mismatch", so it is not folded into `issues`.
    """
    source = get_customer_profile(requested_customer_id, source_path=source_path)
    issues: list[GroundingIssue] = []

    if insight.customer_id != requested_customer_id:
        issues.append(GroundingIssue(
            type="customer_id_mismatch",
            detail=(
                f"customer_id mismatch: insight says '{insight.customer_id}', "
                f"requested '{requested_customer_id}'"
            ),
        ))

    for key_field in insight.key_fields:
        if key_field.field not in source:
            issues.append(GroundingIssue(
                type="unknown_field",
                detail=f"field '{key_field.field}' does not exist in the source record",
            ))
            continue
        if not _values_match(source[key_field.field], key_field.value):
            issues.append(GroundingIssue(
                type="value_mismatch",
                detail=(
                    f"field '{key_field.field}' value mismatch: "
                    f"insight says {key_field.value!r}, source has {source[key_field.field]!r}"
                ),
            ))

    if not _values_match(source["priority_score"], insight.priority_score):
        issues.append(GroundingIssue(
            type="priority_score_mismatch",
            detail=(
                f"priority_score mismatch: insight says {insight.priority_score!r}, "
                f"source has {source['priority_score']!r}"
            ),
        ))

    if not _values_match(source["priority_rank"], insight.priority_rank):
        issues.append(GroundingIssue(
            type="priority_rank_mismatch",
            detail=(
                f"priority_rank mismatch: insight says {insight.priority_rank!r}, "
                f"source has {source['priority_rank']!r}"
            ),
        ))

    return FactCheckResult(
        customer_id=requested_customer_id,
        is_grounded=len(issues) == 0,
        issues=issues,
    )


def _append_audit_log(record: dict[str, Any], path: Path = AUDIT_LOG_PATH) -> None:
    """Append one JSON-lines record for human post-hoc review. Called for
    every attempt AND for the final outcome, regardless of pass/fail."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_grounded_customer_insight(
    customer_id: str,
    llm_client: LLMClient,
    source_path: Path = TOP10_PATH,
    audit_log_path: Path = AUDIT_LOG_PATH,
) -> GroundingOutcome:
    """Step7 grounding policy: generate -> validate -> corrective retry if
    needed -> reject with a fallback message if still ungrounded. Every
    attempt is written to audit_log_path for human review.

    Raises:
        CustomerNotFoundError / CustomerSchemaError / CustomerDataError:
            propagated unchanged -- a missing/malformed source record is a
            system error, not something this policy can retry its way out of.
        InsightGenerationError: if llm_client ever returns something that
            isn't a valid CustomerInsight (on either attempt).
    """
    now = lambda: datetime.now(timezone.utc).isoformat()  # noqa: E731

    insight = generate_customer_insight(customer_id, llm_client, source_path=source_path)
    initial_fact_check = validate_customer_insight(insight, customer_id, source_path=source_path)
    _append_audit_log({
        "timestamp": now(),
        "customer_id": customer_id,
        "attempt": 1,
        "is_grounded": initial_fact_check.is_grounded,
        "issues": [i.model_dump() for i in initial_fact_check.issues],
    }, path=audit_log_path)

    if initial_fact_check.is_grounded:
        outcome = GroundingOutcome(
            customer_id=customer_id,
            status="grounded",
            insight=insight,
            initial_insight=insight,
            initial_fact_check=initial_fact_check,
            final_fact_check=initial_fact_check,
            attempts=1,
        )
        _append_audit_log({"timestamp": now(), "customer_id": customer_id, "final_status": outcome.status}, path=audit_log_path)
        return outcome

    # Corrective retry: feed the specific issues back, not a blind retry.
    profile = get_customer_profile(customer_id, source_path=source_path)
    correction_prompt = build_correction_prompt(
        profile, insight, [issue.detail for issue in initial_fact_check.issues]
    )
    retried_insight = llm_client(correction_prompt)
    if not isinstance(retried_insight, CustomerInsight):
        raise InsightGenerationError(
            f"llm_client returned {type(retried_insight).__name__} on corrective "
            "retry, expected CustomerInsight"
        )

    final_fact_check = validate_customer_insight(retried_insight, customer_id, source_path=source_path)
    _append_audit_log({
        "timestamp": now(),
        "customer_id": customer_id,
        "attempt": 2,
        "is_grounded": final_fact_check.is_grounded,
        "issues": [i.model_dump() for i in final_fact_check.issues],
    }, path=audit_log_path)

    if final_fact_check.is_grounded:
        outcome = GroundingOutcome(
            customer_id=customer_id,
            status="grounded_after_retry",
            insight=retried_insight,
            initial_insight=insight,
            initial_fact_check=initial_fact_check,
            final_fact_check=final_fact_check,
            attempts=2,
        )
    else:
        outcome = GroundingOutcome(
            customer_id=customer_id,
            status="rejected",
            insight=None,
            initial_insight=insight,
            initial_fact_check=initial_fact_check,
            final_fact_check=final_fact_check,
            attempts=2,
            fallback_message=FALLBACK_MESSAGE,
        )

    _append_audit_log({"timestamp": now(), "customer_id": customer_id, "final_status": outcome.status}, path=audit_log_path)
    return outcome


def main() -> None:
    load_env_file()

    if len(sys.argv) > 1:
        customer_id = sys.argv[1]
    else:
        import pandas as pd

        df = pd.read_csv(TOP10_PATH)
        customer_id = df.loc[df["priority_rank"].idxmin(), "customer_id"]

    try:
        llm_client = select_llm_client()
        outcome = generate_grounded_customer_insight(customer_id, llm_client)
    except CustomerDataError as e:
        print(f"Could not load customer data: {e}")
        raise SystemExit(1)
    except InsightGenerationError as e:
        print(f"{e}")
        raise SystemExit(1)

    print(outcome.model_dump_json(indent=2))
    if outcome.status == "rejected":
        print(f"\n{outcome.fallback_message}")


if __name__ == "__main__":
    main()
