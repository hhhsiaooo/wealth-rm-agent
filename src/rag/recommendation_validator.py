"""Phase 3.11 (WHAT): Recommendation output validation.

這是 LLM 寫完 explanation／products 之後的「抓錯」步驟：拿 LLM 的輸出去跟
規則層（誰說了算）和實際檢索到的產品（有哪些可以推薦）比對，抓出捏造或
改動過的內容。這個檔案本身也不呼叫 LLM，只做資料比對。

Mirrors insight/fact_validator.py's role for WHY: checks an LLM-assisted
NextBestAction against the deterministic sources of truth --
ActionSelectionResult (rule layer) and the retrieved Product Knowledge.

Checks (docs/ai_product_design.md, "Recommendation Validation"):
- customer_id / recommended_action / eligible match the rule layer
- every recommended product_id was actually retrieved
- product_name matches the Product Knowledge catalog (not modified)
- sources only reference retrieved products
- no products when retrieval did not run (A1/A5/A6)
- (optional) product risk_level does not exceed the customer's risk_level

Known limitation: the free-text `reason` / `explanation` fields are NOT
verified here (claims inside prose, e.g. a stated risk level, cannot be
checked deterministically without false positives, since prose may mention
both customer and product attributes). Deferred: LLM-as-judge or structured
claims.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from rag.business_rules import ACTION_NAMES, RISK_ORDER  # noqa: E402
from rag.product_knowledge import PRODUCTS, ProductKnowledge  # noqa: E402
from rag.schemas import (  # noqa: E402
    ActionSelectionResult,
    NextBestAction,
    RecommendationCheckResult,
)

SOURCE_PREFIX = "product_knowledge/"


def default_catalog() -> dict[str, ProductKnowledge]:
    return {p.metadata.product_id: p for p in PRODUCTS}


def validate_next_best_action(
    nba: NextBestAction,
    selection: ActionSelectionResult,
    hits: list[dict[str, Any]],
    catalog: dict[str, ProductKnowledge] | None = None,
    customer_risk_level: str | None = None,
) -> RecommendationCheckResult:
    catalog = catalog if catalog is not None else default_catalog()
    retrieved_ids = {h["product_id"] for h in hits}
    issues: list[str] = []

    if nba.customer_id != selection.customer_id:
        issues.append(
            f"customer_id mismatch: NBA says '{nba.customer_id}', "
            f"rule layer has '{selection.customer_id}'"
        )

    if nba.recommended_action != selection.recommended_action:
        issues.append(
            f"recommended_action mismatch: NBA says '{nba.recommended_action}', "
            f"rule layer selected '{selection.recommended_action}'"
        )

    selected = next(
        (a for a in selection.evaluated_actions if a.action_id == selection.recommended_action),
        None,
    )
    if selected is not None and nba.eligible != selected.eligible:
        issues.append(
            f"eligible mismatch: NBA says {nba.eligible}, rule layer says {selected.eligible}"
        )

    expected_name = ACTION_NAMES.get(selection.recommended_action)
    if expected_name is not None and nba.action_name != expected_name:
        issues.append(
            f"action_name mismatch: NBA says '{nba.action_name}', expected '{expected_name}'"
        )

    if not hits and nba.products:
        issues.append("products were recommended although no Product Knowledge was retrieved")

    for product in nba.products:
        if product.product_id not in retrieved_ids:
            issues.append(f"product '{product.product_id}' is not among the retrieved Product Knowledge")
            continue
        known = catalog.get(product.product_id)
        if known is not None and product.product_name != known.content.product_name:
            issues.append(
                f"product_name mismatch for '{product.product_id}': NBA says "
                f"'{product.product_name}', catalog has '{known.content.product_name}'"
            )
        if (
            known is not None
            and customer_risk_level in RISK_ORDER
            and RISK_ORDER[known.metadata.risk_level] > RISK_ORDER[customer_risk_level]
        ):
            issues.append(
                f"product '{product.product_id}' risk '{known.metadata.risk_level}' exceeds "
                f"customer risk '{customer_risk_level}'"
            )

    for source in nba.sources:
        product_id = source[len(SOURCE_PREFIX):] if source.startswith(SOURCE_PREFIX) else source
        if product_id not in retrieved_ids:
            issues.append(f"source '{source}' does not reference a retrieved product")

    return RecommendationCheckResult(
        customer_id=selection.customer_id,
        is_grounded=not issues,
        issues=issues,
    )
