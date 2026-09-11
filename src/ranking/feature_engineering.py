"""Step 3: Derived feature engineering.

Reads data/synthetic/customers_raw.csv, preserves every raw field as-is, and
appends only the 5 Derived features defined in
docs/ai_product_data_definition.xlsx (Data_Specification sheet):

    total_aum         = twd_demand_balance + twd_td_balance
                         + fx_demand_balance + fx_td_balance + investment_balance
    twd_demand_ratio  = twd_demand_balance / total_aum
    td_ratio          = (twd_td_balance + fx_td_balance) / total_aum
    fx_ratio          = (fx_demand_balance + fx_td_balance) / total_aum
    investment_ratio  = investment_balance / total_aum

No feature-level scores, Sub-Scores, or Priority Score are computed here.

Ratios are stored as fractions (0.0-1.0), matching the literal formulas in
the spec (no x100), consistent with the ratio targets already used in
generate_data.py's ARCHETYPE_CONFIG.

Edge case: when total_aum == 0 (all 5 balances are 0), all 4 ratio fields
are set to 0 (confirmed business rule, not a spec formula).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "synthetic"
RAW_PATH = DATA_DIR / "customers_raw.csv"
OUTPUT_PATH = DATA_DIR / "customers_features.csv"

BALANCE_COLUMNS = [
    "twd_demand_balance",
    "twd_td_balance",
    "fx_demand_balance",
    "fx_td_balance",
    "investment_balance",
]

DERIVED_COLUMNS = [
    "total_aum",
    "twd_demand_ratio",
    "td_ratio",
    "fx_ratio",
    "investment_ratio",
]


def load_raw_data(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def compute_total_aum(df: pd.DataFrame) -> pd.Series:
    return df[BALANCE_COLUMNS].sum(axis=1)


def _safe_ratio(numerator: pd.Series, total_aum: pd.Series) -> pd.Series:
    """numerator / total_aum, with 0 when total_aum == 0 (avoids division by zero)."""
    return np.where(total_aum > 0, numerator / total_aum, 0.0)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    total_aum = compute_total_aum(df)
    df["total_aum"] = total_aum
    df["twd_demand_ratio"] = _safe_ratio(df["twd_demand_balance"], total_aum)
    df["td_ratio"] = _safe_ratio(df["twd_td_balance"] + df["fx_td_balance"], total_aum)
    df["fx_ratio"] = _safe_ratio(df["fx_demand_balance"] + df["fx_td_balance"], total_aum)
    df["investment_ratio"] = _safe_ratio(df["investment_balance"], total_aum)

    return df


def validate_features(df_raw: pd.DataFrame, df_features: pd.DataFrame) -> list[str]:
    """Return a list of validation failure messages (empty if all pass)."""
    errors: list[str] = []

    if len(df_features) != len(df_raw):
        errors.append(f"row count = {len(df_features)}, expected {len(df_raw)}")

    missing_raw = set(df_raw.columns) - set(df_features.columns)
    if missing_raw:
        errors.append(f"raw columns missing from output: {missing_raw}")

    missing_derived = set(DERIVED_COLUMNS) - set(df_features.columns)
    if missing_derived:
        errors.append(f"derived columns missing from output: {missing_derived}")

    if "total_aum" in df_features.columns:
        expected_total_aum = compute_total_aum(df_features)
        if not np.array_equal(df_features["total_aum"].to_numpy(), expected_total_aum.to_numpy()):
            errors.append("total_aum is not consistent with the sum of its 5 balance components")

        if (df_features["total_aum"] < 0).any():
            errors.append("total_aum has negative values")

    ratio_checks = {
        "twd_demand_ratio": df_features["twd_demand_balance"],
        "td_ratio": df_features["twd_td_balance"] + df_features["fx_td_balance"],
        "fx_ratio": df_features["fx_demand_balance"] + df_features["fx_td_balance"],
        "investment_ratio": df_features["investment_balance"],
    }
    for col, numerator in ratio_checks.items():
        if col not in df_features.columns or "total_aum" not in df_features.columns:
            continue  # already reported via missing_derived
        values = df_features[col]

        if not values.between(0.0, 1.0).all():
            errors.append(f"{col} has values outside [0, 1]")

        expected = _safe_ratio(numerator, df_features["total_aum"])
        if not np.allclose(values.to_numpy(), expected, atol=1e-9):
            errors.append(f"{col} does not match numerator/total_aum")

        zero_aum = df_features["total_aum"] == 0
        if zero_aum.any() and not (values[zero_aum] == 0).all():
            errors.append(f"{col} is not 0 when total_aum == 0")

    return errors


def main() -> None:
    df_raw = load_raw_data()
    df_features = add_derived_features(df_raw)

    errors = validate_features(df_raw, df_features)
    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    df_features.to_csv(OUTPUT_PATH, index=False)

    print(f"Generated {len(df_features)} rows -> {OUTPUT_PATH}")
    print("Validation: all checks passed")
    print("\nDerived feature summary:")
    print(df_features[DERIVED_COLUMNS].describe().round(4))


if __name__ == "__main__":
    main()
