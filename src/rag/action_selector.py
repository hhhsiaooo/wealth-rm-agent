"""Phase 3.2 (WHAT): Action Selection.

同樣不呼叫 LLM：business_rules.py 算出「哪些行動成立」，這個檔案只是照
固定優先順序（A2>A3>A4>A5>A1>A6）從成立的行動裡選一個當作最終建議。

Every customer in this project's synthetic Top10 population satisfies more
than one Eligibility rule at once (confirmed empirically during the Phase 3
readiness review), so a deterministic tie-break is not an edge case here --
it fires for every customer. PRIORITY_ORDER is a fixed constant; the LLM
never decides it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from rag.business_rules import evaluate_actions  # noqa: E402
from rag.schemas import ActionSelectionResult  # noqa: E402
from tools.customer_tool import TOP10_PATH, get_customer_profile  # noqa: E402

PRIORITY_ORDER = ["A2", "A3", "A4", "A5", "A1", "A6"]


def select_action_for_profile(customer: dict[str, Any]) -> ActionSelectionResult:
    """Pure version: takes an already-fetched customer profile dict.
    evaluate_actions() guarantees A6 is eligible whenever A1-A5 are all
    False, so exactly one action in PRIORITY_ORDER is always eligible."""
    evaluated = evaluate_actions(customer)
    eligible_ids = {a.action_id for a in evaluated if a.eligible}

    recommended = next(action_id for action_id in PRIORITY_ORDER if action_id in eligible_ids)

    return ActionSelectionResult(
        customer_id=customer["customer_id"],
        evaluated_actions=evaluated,
        recommended_action=recommended,
    )


def select_action(customer_id: str, source_path: Path = TOP10_PATH) -> ActionSelectionResult:
    """Fetch the customer's 43-field record and select the recommended action.

    Raises:
        CustomerNotFoundError / CustomerSchemaError / CustomerDataError:
            propagated unchanged from get_customer_profile().
    """
    customer = get_customer_profile(customer_id, source_path=source_path)
    return select_action_for_profile(customer)
