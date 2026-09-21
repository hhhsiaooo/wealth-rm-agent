from __future__ import annotations

import pandas as pd
import pytest

from rag.action_selector import PRIORITY_ORDER, select_action, select_action_for_profile
from rag.schemas import ActionSelectionResult
from tools.customer_tool import EXPECTED_FIELDS, TOP10_PATH, CustomerNotFoundError


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


def write_csv(tmp_path, rows: list[dict]):
    path = tmp_path / "priority_customers_top10.csv"
    pd.DataFrame(rows, columns=EXPECTED_FIELDS).to_csv(path, index=False)
    return path


def test_priority_order_matches_spec():
    assert PRIORITY_ORDER == ["A2", "A3", "A4", "A5", "A1", "A6"]


# -- select_action_for_profile: priority resolution --------------------------

def test_picks_a2_over_a1_and_a3_when_all_eligible():
    customer = make_customer(
        has_fund=0, investment_intent_score=92, risk_level="High",
        investment_ratio=0.0, twd_td_balance=8_000_000,
    )
    result = select_action_for_profile(customer)
    assert isinstance(result, ActionSelectionResult)
    assert result.recommended_action == "A2"


def test_picks_a3_over_a4_a5_a1_when_a2_not_eligible():
    # has_fund=1 disqualifies A2; twd_td_balance>0 qualifies A3;
    # fx_demand_balance>0 qualifies A4; investment_balance/ratio qualify A5;
    # investment_intent_score/ratio qualify A1.
    customer = make_customer(
        has_fund=1, twd_td_balance=1, fx_demand_balance=1,
        investment_balance=1, investment_ratio=0.5,
        investment_intent_score=80,
    )
    result = select_action_for_profile(customer)
    assert result.recommended_action == "A3"


def test_picks_a4_over_a5_and_a1_when_a2_a3_not_eligible():
    customer = make_customer(
        has_fund=1, twd_td_balance=0, fx_td_balance=0, fx_demand_balance=1,
        investment_balance=1, investment_ratio=0.5, investment_intent_score=80,
    )
    result = select_action_for_profile(customer)
    assert result.recommended_action == "A4"


def test_picks_a5_over_a1_when_only_those_two_eligible():
    customer = make_customer(
        investment_balance=1, investment_ratio=0.5, investment_intent_score=60,
    )
    result = select_action_for_profile(customer)
    assert result.recommended_action == "A5"


def test_picks_a1_when_only_a1_eligible():
    customer = make_customer(investment_intent_score=60, investment_ratio=0.1)
    result = select_action_for_profile(customer)
    assert result.recommended_action == "A1"


def test_falls_back_to_a6_when_nothing_eligible():
    customer = make_customer()
    result = select_action_for_profile(customer)
    assert result.recommended_action == "A6"


def test_result_includes_all_six_evaluated_actions():
    result = select_action_for_profile(make_customer())
    assert len(result.evaluated_actions) == 6
    assert result.customer_id == "C000001"


# -- select_action: integration with get_customer_profile --------------------

def test_select_action_reads_from_source_path(tmp_path):
    path = write_csv(tmp_path, [make_customer(
        customer_id="C000104", has_fund=0, investment_intent_score=92,
        risk_level="High", twd_td_balance=8_000_000,
    )])
    result = select_action("C000104", source_path=path)
    assert result.recommended_action == "A2"


def test_select_action_unknown_customer_id_propagates(tmp_path):
    path = write_csv(tmp_path, [make_customer(customer_id="C000104")])
    with pytest.raises(CustomerNotFoundError):
        select_action("C999999", source_path=path)


# -- end-to-end on actual WHO/WHY Top10 output --------------------------------

@pytest.mark.skipif(not TOP10_PATH.exists(), reason="priority_customers_top10.csv not generated yet")
def test_end_to_end_every_top10_customer_gets_a_recommendation():
    df = pd.read_csv(TOP10_PATH)
    for customer_id in df["customer_id"]:
        result = select_action(customer_id)
        assert result.recommended_action in PRIORITY_ORDER
        assert result.customer_id == customer_id


@pytest.mark.skipif(not TOP10_PATH.exists(), reason="priority_customers_top10.csv not generated yet")
def test_end_to_end_recommendation_distribution_matches_readiness_review():
    """Sanity check tying back to the Phase 3 readiness review: with the
    current synthetic Top10 population, every customer ends up on A2 or A3
    (confirmed empirically before implementation started)."""
    df = pd.read_csv(TOP10_PATH)
    recommended = [select_action(cid).recommended_action for cid in df["customer_id"]]
    assert set(recommended) <= {"A2", "A3"}
    assert recommended.count("A2") == 7
    assert recommended.count("A3") == 5
