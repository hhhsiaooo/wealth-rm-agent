import numpy as np
import pandas as pd
import pytest

from ranking.feature_engineering import (
    BALANCE_COLUMNS,
    DERIVED_COLUMNS,
    add_derived_features,
    compute_total_aum,
    load_raw_data,
    validate_features,
    RAW_PATH,
)


def make_raw_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal raw-like DataFrame with the 5 balance columns plus
    customer_id, filling any unspecified balance with 0."""
    records = []
    for i, row in enumerate(rows):
        record = {"customer_id": f"C{i:06d}"}
        for col in BALANCE_COLUMNS:
            record[col] = row.get(col, 0)
        records.append(record)
    return pd.DataFrame(records)


def test_total_aum_sums_five_balances():
    df = make_raw_df([
        {
            "twd_demand_balance": 1_000_000,
            "twd_td_balance": 2_000_000,
            "fx_demand_balance": 500_000,
            "fx_td_balance": 500_000,
            "investment_balance": 1_000_000,
        }
    ])
    result = add_derived_features(df)
    assert result.loc[0, "total_aum"] == 5_000_000


def test_ratio_formulas_are_correct():
    df = make_raw_df([
        {
            "twd_demand_balance": 4_000_000,
            "twd_td_balance": 2_000_000,
            "fx_demand_balance": 1_000_000,
            "fx_td_balance": 1_000_000,
            "investment_balance": 2_000_000,
        }
    ])
    # total_aum = 10,000,000
    result = add_derived_features(df)
    row = result.loc[0]

    assert row["total_aum"] == 10_000_000
    assert row["twd_demand_ratio"] == pytest.approx(4_000_000 / 10_000_000)
    assert row["td_ratio"] == pytest.approx((2_000_000 + 1_000_000) / 10_000_000)
    assert row["fx_ratio"] == pytest.approx((1_000_000 + 1_000_000) / 10_000_000)
    assert row["investment_ratio"] == pytest.approx(2_000_000 / 10_000_000)


def test_total_aum_zero_sets_all_ratios_to_zero_without_error():
    df = make_raw_df([{}])  # all 5 balances default to 0
    result = add_derived_features(df)
    row = result.loc[0]

    assert row["total_aum"] == 0
    assert row["twd_demand_ratio"] == 0
    assert row["td_ratio"] == 0
    assert row["fx_ratio"] == 0
    assert row["investment_ratio"] == 0
    ratio_values = row[["twd_demand_ratio", "td_ratio", "fx_ratio", "investment_ratio"]].astype(float)
    assert not np.isnan(ratio_values).any()


def test_single_nonzero_component_gives_ratio_one():
    df = make_raw_df([{"investment_balance": 3_000_000}])
    result = add_derived_features(df)
    row = result.loc[0]

    assert row["investment_ratio"] == pytest.approx(1.0)
    assert row["twd_demand_ratio"] == 0
    assert row["td_ratio"] == 0
    assert row["fx_ratio"] == 0


def test_raw_columns_are_preserved():
    df = make_raw_df([{"twd_demand_balance": 1_000_000}])
    result = add_derived_features(df)
    for col in df.columns:
        assert col in result.columns
    pd.testing.assert_series_equal(result["customer_id"], df["customer_id"])


def test_output_has_no_score_or_priority_columns():
    df = make_raw_df([{"twd_demand_balance": 1_000_000}])
    result = add_derived_features(df)

    forbidden_substrings = ["_score", "priority"]
    for col in result.columns:
        for token in forbidden_substrings:
            assert token not in col.lower(), f"unexpected scoring column found: {col}"

    assert set(result.columns) == set(df.columns) | set(DERIVED_COLUMNS)


def test_row_count_is_preserved():
    df = make_raw_df([{"twd_demand_balance": i * 100_000} for i in range(50)])
    result = add_derived_features(df)
    assert len(result) == len(df)


def test_validate_features_passes_on_correct_output():
    df = make_raw_df([
        {"twd_demand_balance": 1_000_000, "investment_balance": 2_000_000},
        {},  # zero total_aum edge case
        {"fx_td_balance": 500_000, "twd_td_balance": 500_000},
    ])
    result = add_derived_features(df)
    errors = validate_features(df, result)
    assert errors == []


def test_validate_features_detects_broken_total_aum():
    df = make_raw_df([{"twd_demand_balance": 1_000_000, "investment_balance": 1_000_000}])
    result = add_derived_features(df)
    result["total_aum"] = result["total_aum"] + 999  # corrupt it

    errors = validate_features(df, result)
    assert any("total_aum" in e for e in errors)


def test_validate_features_detects_missing_derived_column():
    df = make_raw_df([{"twd_demand_balance": 1_000_000}])
    result = add_derived_features(df).drop(columns=["fx_ratio"])

    errors = validate_features(df, result)
    assert any("fx_ratio" in e for e in errors)


@pytest.mark.skipif(not RAW_PATH.exists(), reason="customers_raw.csv not generated yet")
def test_end_to_end_on_actual_raw_data():
    df_raw = load_raw_data()
    df_features = add_derived_features(df_raw)

    errors = validate_features(df_raw, df_features)
    assert errors == []
    assert len(df_features) == len(df_raw)
    assert (df_features["total_aum"] == compute_total_aum(df_raw)).all()
