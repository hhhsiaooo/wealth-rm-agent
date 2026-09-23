from __future__ import annotations

from rag.product_knowledge import PRODUCTS
from rag.recommendation_validator import validate_next_best_action
from rag.schemas import (
    ActionEligibility,
    ActionSelectionResult,
    NextBestAction,
    ProductRecommendation,
    RecommendationCheckResult,
)

FUND_007_NAME = next(p.content.product_name for p in PRODUCTS if p.metadata.product_id == "FUND_007")
FUND_008_NAME = next(p.content.product_name for p in PRODUCTS if p.metadata.product_id == "FUND_008")


def make_selection(recommended="A2", customer_id="C000104") -> ActionSelectionResult:
    return ActionSelectionResult(
        customer_id=customer_id,
        evaluated_actions=[
            ActionEligibility(action_id=a, eligible=(a == recommended), matched_conditions=["x"] if a == recommended else [])
            for a in ["A1", "A2", "A3", "A4", "A5", "A6"]
        ],
        recommended_action=recommended,
    )


def make_hits(*ids: str) -> list[dict]:
    return [{"product_id": i, "document": "", "metadata": {}, "distance": 0.1} for i in ids]


def make_nba(**overrides) -> NextBestAction:
    data = dict(
        customer_id="C000104",
        recommended_action="A2",
        action_name="基金產品介紹",
        eligible=True,
        explanation="說明",
        products=[ProductRecommendation(product_id="FUND_007", product_name=FUND_007_NAME, reason="r")],
        sources=["product_knowledge/FUND_007"],
    )
    data.update(overrides)
    return NextBestAction(**data)


def test_correct_nba_is_grounded():
    result = validate_next_best_action(make_nba(), make_selection(), make_hits("FUND_007", "FUND_008"))
    assert isinstance(result, RecommendationCheckResult)
    assert result.is_grounded is True
    assert result.issues == []


def test_flags_changed_recommended_action():
    nba = make_nba(recommended_action="A3", action_name="定存資產配置討論/產品介紹")
    result = validate_next_best_action(nba, make_selection("A2"), make_hits("FUND_007"))
    assert result.is_grounded is False
    assert any("recommended_action mismatch" in i for i in result.issues)


def test_flags_changed_eligible():
    result = validate_next_best_action(make_nba(eligible=False), make_selection(), make_hits("FUND_007"))
    assert any("eligible mismatch" in i for i in result.issues)


def test_flags_wrong_customer_id():
    result = validate_next_best_action(make_nba(customer_id="C999999"), make_selection(), make_hits("FUND_007"))
    assert any("customer_id mismatch" in i for i in result.issues)


def test_flags_product_not_retrieved():
    nba = make_nba(
        products=[ProductRecommendation(product_id="FUND_008", product_name=FUND_008_NAME, reason="r")],
        sources=["product_knowledge/FUND_008"],
    )
    result = validate_next_best_action(nba, make_selection(), make_hits("FUND_007"))
    assert any("FUND_008" in i and "not among the retrieved" in i for i in result.issues)


def test_flags_invented_product_id():
    nba = make_nba(
        products=[ProductRecommendation(product_id="FUND_999", product_name="幻覺基金", reason="r")],
        sources=["product_knowledge/FUND_999"],
    )
    result = validate_next_best_action(nba, make_selection(), make_hits("FUND_007"))
    assert result.is_grounded is False


def test_flags_modified_product_name():
    nba = make_nba(products=[ProductRecommendation(product_id="FUND_007", product_name="被改過的名字", reason="r")])
    result = validate_next_best_action(nba, make_selection(), make_hits("FUND_007"))
    assert any("product_name mismatch" in i for i in result.issues)


def test_flags_source_not_retrieved():
    nba = make_nba(sources=["product_knowledge/FUND_008"])
    result = validate_next_best_action(nba, make_selection(), make_hits("FUND_007"))
    assert any("source" in i for i in result.issues)


def test_flags_products_when_nothing_was_retrieved():
    result = validate_next_best_action(make_nba(), make_selection("A1"), hits=[])
    assert any("no Product Knowledge was retrieved" in i for i in result.issues)


def test_no_products_and_no_hits_is_grounded_for_non_retrieval_action():
    nba = make_nba(recommended_action="A1", action_name="投資需求訪談", products=[], sources=[])
    result = validate_next_best_action(nba, make_selection("A1"), hits=[])
    assert result.is_grounded is True


def test_multiple_issues_are_all_reported():
    nba = make_nba(
        recommended_action="A3", eligible=False,
        products=[ProductRecommendation(product_id="FUND_999", product_name="x", reason="r")],
        sources=["product_knowledge/FUND_999"],
    )
    result = validate_next_best_action(nba, make_selection("A2"), make_hits("FUND_007"))
    assert len(result.issues) >= 4


def test_flags_product_riskier_than_customer():
    # FUND_007 is High risk; a Medium customer must not be shown it.
    result = validate_next_best_action(
        make_nba(), make_selection(), make_hits("FUND_007"), customer_risk_level="Medium"
    )
    assert any("exceeds customer risk" in i for i in result.issues)


def test_product_within_customer_risk_passes():
    result = validate_next_best_action(
        make_nba(), make_selection(), make_hits("FUND_007"), customer_risk_level="High"
    )
    assert result.is_grounded is True
