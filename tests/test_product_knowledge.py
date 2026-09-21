from __future__ import annotations

from rag.product_knowledge import PRODUCTS, load_products, load_products_by_category

VALID_CATEGORIES = {"Fund", "TD", "FX"}
VALID_RISK_LEVELS = {"Low", "Medium", "High"}


def test_total_product_count_is_30():
    assert len(PRODUCTS) == 30


def test_ten_products_per_category():
    for category in VALID_CATEGORIES:
        assert len(load_products_by_category(category)) == 10


def test_product_ids_are_unique():
    ids = [p.metadata.product_id for p in PRODUCTS]
    assert len(ids) == len(set(ids))


def test_all_categories_and_risk_levels_are_valid():
    for p in PRODUCTS:
        assert p.metadata.product_category in VALID_CATEGORIES
        assert p.metadata.risk_level in VALID_RISK_LEVELS


def test_metadata_defaults_are_consistent():
    for p in PRODUCTS:
        assert p.metadata.product_status == "active"
        assert p.metadata.version == "1.0"
        assert p.metadata.source == "synthetic_poc"


def test_load_products_returns_a_copy_not_the_shared_list():
    result = load_products()
    result.clear()
    assert len(PRODUCTS) == 30  # original module-level list untouched


def test_to_chunk_text_contains_product_name_and_cautions():
    product = PRODUCTS[0]
    chunk = product.to_chunk_text()
    assert product.content.product_name in chunk
    assert product.content.sales_cautions in chunk
    assert len(chunk) > 0


def test_to_flat_metadata_has_all_eight_fields_as_strings():
    product = PRODUCTS[0]
    flat = product.to_flat_metadata()
    expected_keys = {
        "product_id", "product_category", "risk_level", "currency",
        "product_status", "effective_date", "version", "source",
    }
    assert set(flat.keys()) == expected_keys
    assert all(isinstance(v, str) for v in flat.values())


def test_fund_products_include_medium_and_high_risk_for_a2_matching():
    """A2's rule requires customer risk_level in {Medium, High}; the Fund
    catalogue must actually have Medium/High risk products for retrieval to
    find a plausible match."""
    fund_risk_levels = {p.metadata.risk_level for p in load_products_by_category("Fund")}
    assert "Medium" in fund_risk_levels or "High" in fund_risk_levels
