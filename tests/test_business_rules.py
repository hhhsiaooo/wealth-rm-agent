from __future__ import annotations

import pytest

from rag.business_rules import (
    ACTION_NAMES,
    check_a1,
    check_a2,
    check_a3,
    check_a4,
    check_a5,
    check_a6,
    evaluate_actions,
)
from tools.customer_tool import EXPECTED_FIELDS


def make_customer(**overrides) -> dict:
    customer = {field: 0 for field in EXPECTED_FIELDS}
    customer.update({
        "customer_id": "C000001",
        "risk_level": "Low",
        "has_fund": 0,
        "investment_intent_score": 0,
        "investment_ratio": 0.0,
        "twd_td_balance": 0,
        "fx_td_balance": 0,
        "fx_demand_balance": 0,
        "investment_balance": 0,
    })
    customer.update(overrides)
    return customer


# -- A1: investment_intent_score >= 50 AND investment_ratio < 0.30 ----------

@pytest.mark.parametrize("intent,ratio,expected", [
    (50, 0.0, True),
    (49, 0.0, False),
    (100, 0.2999, True),
    (100, 0.30, False),
    (100, 0.31, False),
])
def test_check_a1_boundaries(intent, ratio, expected):
    eligible, conditions = check_a1(make_customer(investment_intent_score=intent, investment_ratio=ratio))
    assert eligible is expected
    assert bool(conditions) == expected


def test_check_a1_matched_conditions_content():
    eligible, conditions = check_a1(make_customer(investment_intent_score=92, investment_ratio=0.1))
    assert eligible is True
    assert any("investment_intent_score" in c for c in conditions)
    assert any("investment_ratio" in c for c in conditions)


# -- A2: has_fund=0 AND investment_intent_score>=70 AND risk_level in {Medium,High}

@pytest.mark.parametrize("has_fund,intent,risk_level,expected", [
    (0, 70, "Medium", True),
    (0, 70, "High", True),
    (0, 69, "Medium", False),
    (0, 70, "Low", False),
    (1, 70, "Medium", False),
    (0, 100, "medium", False),  # case-sensitive: lowercase must NOT match
])
def test_check_a2_boundaries(has_fund, intent, risk_level, expected):
    eligible, _ = check_a2(make_customer(has_fund=has_fund, investment_intent_score=intent, risk_level=risk_level))
    assert eligible is expected


# -- A3: twd_td_balance > 0 OR fx_td_balance > 0 ------------------------------

@pytest.mark.parametrize("twd_td,fx_td,expected", [
    (0, 0, False),
    (1, 0, True),
    (0, 1, True),
    (1, 1, True),
])
def test_check_a3_boundaries(twd_td, fx_td, expected):
    eligible, conditions = check_a3(make_customer(twd_td_balance=twd_td, fx_td_balance=fx_td))
    assert eligible is expected
    assert bool(conditions) == expected


def test_check_a3_matched_conditions_only_lists_satisfied_side():
    eligible, conditions = check_a3(make_customer(twd_td_balance=8_000_000, fx_td_balance=0))
    assert eligible is True
    assert len(conditions) == 1
    assert "twd_td_balance" in conditions[0]


# -- A4: fx_demand_balance > 0 OR fx_td_balance > 0 --------------------------

@pytest.mark.parametrize("fx_demand,fx_td,expected", [
    (0, 0, False),
    (1, 0, True),
    (0, 1, True),
])
def test_check_a4_boundaries(fx_demand, fx_td, expected):
    eligible, _ = check_a4(make_customer(fx_demand_balance=fx_demand, fx_td_balance=fx_td))
    assert eligible is expected


# -- A5: investment_balance > 0 AND investment_ratio >= 0.30 -----------------

@pytest.mark.parametrize("balance,ratio,expected", [
    (0, 0.50, False),          # balance=0 disqualifies regardless of ratio
    (100, 0.2999, False),
    (100, 0.30, True),
    (100, 1.0, True),
])
def test_check_a5_boundaries(balance, ratio, expected):
    eligible, _ = check_a5(make_customer(investment_balance=balance, investment_ratio=ratio))
    assert eligible is expected


# -- A6: 不符合 A1~A5 任一條件 -------------------------------------------------

def test_check_a6_eligible_when_none_of_a1_to_a5_eligible():
    eligible, conditions = check_a6([False, False, False, False, False])
    assert eligible is True
    assert conditions != []


@pytest.mark.parametrize("others", [
    [True, False, False, False, False],
    [False, False, False, False, True],
    [True, True, True, True, True],
])
def test_check_a6_not_eligible_when_any_other_eligible(others):
    eligible, conditions = check_a6(others)
    assert eligible is False
    assert conditions == []


# -- evaluate_actions ---------------------------------------------------------

def test_evaluate_actions_returns_all_six_in_order():
    result = evaluate_actions(make_customer())
    assert [a.action_id for a in result] == ["A1", "A2", "A3", "A4", "A5", "A6"]


def test_evaluate_actions_no_signals_falls_back_to_a6():
    result = evaluate_actions(make_customer())  # all defaults are "no signal"
    by_id = {a.action_id: a for a in result}
    assert by_id["A6"].eligible is True
    for action_id in ["A1", "A2", "A3", "A4", "A5"]:
        assert by_id[action_id].eligible is False


def test_evaluate_actions_multiple_eligible_at_once():
    customer = make_customer(
        has_fund=0, investment_intent_score=92, risk_level="High",
        investment_ratio=0.0, twd_td_balance=8_000_000,
    )
    result = evaluate_actions(customer)
    by_id = {a.action_id: a for a in result}
    assert by_id["A1"].eligible is True
    assert by_id["A2"].eligible is True
    assert by_id["A3"].eligible is True
    assert by_id["A6"].eligible is False  # must not fire when others are eligible


def test_action_names_cover_all_six_ids():
    assert set(ACTION_NAMES.keys()) == {"A1", "A2", "A3", "A4", "A5", "A6"}
