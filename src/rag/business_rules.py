"""Phase 3.1 (WHAT): Eligibility / Business Rules.

這個檔案負責「六個候選行動（A1～A6）哪些成立」，全部是 if/else 數值比較，
完全不呼叫 LLM，也不需要 AI 判斷——這是整條 WHAT 流程裡最「死板」的一層，
故意設計成這樣，因為要選哪個行動必須可重現、可解釋、可稽核。

Each check_aN() is a pure function: customer dict in, (eligible, matched_conditions)
out. No LLM, no rule engine -- transcribed 1:1 from docs/ai_product_design.md
"五、Task／3. Next Best Action／Eligibility / Business Rules". Do not invent
additional business rules; if a rule looks ambiguous or inconsistent with the
data, report it instead of changing it here.

matched_conditions lists the sub-conditions that were actually satisfied when
eligible is True; it is empty when eligible is False (nothing "matched").
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from rag.schemas import ActionEligibility  # noqa: E402

ACTION_NAMES = {
    "A1": "投資需求訪談",
    "A2": "基金產品介紹",
    "A3": "定存資產配置討論/產品介紹",
    "A4": "外幣資產配置討論/產品介紹",
    "A5": "既有投資組合檢視",
    "A6": "暫不聯繫",
}

# Only these actions trigger Product Knowledge retrieval (RAG); value is the
# Knowledge Base product_category to search. A1/A5/A6 get an LLM explanation
# without retrieval (docs/ai_product_design.md, Task 3 / Output).
RETRIEVAL_CATEGORY_BY_ACTION = {"A2": "Fund", "A3": "TD", "A4": "FX"}

A2_ELIGIBLE_RISK_LEVELS = {"Medium", "High"}

# Product-risk suitability filter: not in the original spec, added at the
# product owner's request after the Phase 3 readiness review and confirmed
# (docs/ai_product_design.md, "Eligibility / Business Rules"). A product is
# retrievable for a customer only if its risk_level does not exceed the
# customer's risk_level (Low < Medium < High). Applied as a metadata
# pre-filter in retrieval.
RISK_ORDER = {"Low": 0, "Medium": 1, "High": 2}


def allowed_product_risk_levels(customer_risk_level: str) -> list[str]:
    """Product risk levels a customer may be shown, lowest first."""
    limit = RISK_ORDER[customer_risk_level]
    return [level for level, rank in sorted(RISK_ORDER.items(), key=lambda kv: kv[1]) if rank <= limit]


def check_a1(customer: dict[str, Any]) -> tuple[bool, list[str]]:
    """A1 投資需求訪談: investment_intent_score >= 50 AND investment_ratio < 0.30"""
    intent = customer["investment_intent_score"]
    ratio = customer["investment_ratio"]
    eligible = intent >= 50 and ratio < 0.30
    if not eligible:
        return False, []
    return True, [
        f"investment_intent_score={intent} >= 50",
        f"investment_ratio={ratio} < 0.30",
    ]


def check_a2(customer: dict[str, Any]) -> tuple[bool, list[str]]:
    """A2 基金產品介紹: has_fund = 0 AND investment_intent_score >= 70
    AND risk_level IN {Medium, High}"""
    has_fund = customer["has_fund"]
    intent = customer["investment_intent_score"]
    risk_level = customer["risk_level"]
    eligible = has_fund == 0 and intent >= 70 and risk_level in A2_ELIGIBLE_RISK_LEVELS
    if not eligible:
        return False, []
    return True, [
        f"has_fund={has_fund}",
        f"investment_intent_score={intent} >= 70",
        f"risk_level={risk_level!r} in {sorted(A2_ELIGIBLE_RISK_LEVELS)}",
    ]


def check_a3(customer: dict[str, Any]) -> tuple[bool, list[str]]:
    """A3 定存資產配置討論/產品介紹: twd_td_balance > 0 OR fx_td_balance > 0"""
    twd_td = customer["twd_td_balance"]
    fx_td = customer["fx_td_balance"]
    conditions = []
    if twd_td > 0:
        conditions.append(f"twd_td_balance={twd_td} > 0")
    if fx_td > 0:
        conditions.append(f"fx_td_balance={fx_td} > 0")
    return bool(conditions), conditions


def check_a4(customer: dict[str, Any]) -> tuple[bool, list[str]]:
    """A4 外幣資產配置討論/產品介紹: fx_demand_balance > 0 OR fx_td_balance > 0"""
    fx_demand = customer["fx_demand_balance"]
    fx_td = customer["fx_td_balance"]
    conditions = []
    if fx_demand > 0:
        conditions.append(f"fx_demand_balance={fx_demand} > 0")
    if fx_td > 0:
        conditions.append(f"fx_td_balance={fx_td} > 0")
    return bool(conditions), conditions


def check_a5(customer: dict[str, Any]) -> tuple[bool, list[str]]:
    """A5 既有投資組合檢視: investment_balance > 0 AND investment_ratio >= 0.30"""
    balance = customer["investment_balance"]
    ratio = customer["investment_ratio"]
    eligible = balance > 0 and ratio >= 0.30
    if not eligible:
        return False, []
    return True, [
        f"investment_balance={balance} > 0",
        f"investment_ratio={ratio} >= 0.30",
    ]


def check_a6(other_eligible: list[bool]) -> tuple[bool, list[str]]:
    """A6 暫不聯繫: 不符合 A1～A5 任一條件"""
    eligible = not any(other_eligible)
    conditions = ["不符合 A1～A5 任一條件"] if eligible else []
    return eligible, conditions


def evaluate_actions(customer: dict[str, Any]) -> list[ActionEligibility]:
    """Evaluate all six actions for one customer, in A1..A6 order."""
    a1_eligible, a1_conditions = check_a1(customer)
    a2_eligible, a2_conditions = check_a2(customer)
    a3_eligible, a3_conditions = check_a3(customer)
    a4_eligible, a4_conditions = check_a4(customer)
    a5_eligible, a5_conditions = check_a5(customer)
    a6_eligible, a6_conditions = check_a6(
        [a1_eligible, a2_eligible, a3_eligible, a4_eligible, a5_eligible]
    )

    results = [
        ("A1", a1_eligible, a1_conditions),
        ("A2", a2_eligible, a2_conditions),
        ("A3", a3_eligible, a3_conditions),
        ("A4", a4_eligible, a4_conditions),
        ("A5", a5_eligible, a5_conditions),
        ("A6", a6_eligible, a6_conditions),
    ]
    return [
        ActionEligibility(action_id=action_id, eligible=eligible, matched_conditions=conditions)
        for action_id, eligible, conditions in results
    ]
