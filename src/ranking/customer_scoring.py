"""Step 4: Feature-level scoring.

Reads data/synthetic/customers_features.csv, preserves every existing
column, and appends the 9 feature-level score columns defined in
docs/ai_product_data_definition.xlsx (Score_Mapping sheet):

    occupation_score, job_level_score, has_fund_score, total_aum_score,
    twd_demand_ratio_score, investment_ratio_score, inflow_1m_score,
    fund_view_count_1m_score, has_abandoned_application_1m_score

No Sub-Scores, Priority Score, or Priority Rank are computed here (Step 5).

Every mapping below is transcribed literally from the Score_Mapping sheet,
including its boundary operators (">", "<", "<=", ">="). Ratio fields
(twd_demand_ratio, investment_ratio) are read as fractions (0.0-1.0), matching
how they were stored in Step 3, so the sheet's "10%"/"40%"/"60%" thresholds
are expressed here as 0.10/0.40/0.60.

If any input value cannot be mapped by its rule (wrong type, out of the
[0, 1]/non-negative domain, unknown category), this module raises a
ValueError instead of silently assigning a default score.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "synthetic"
FEATURES_PATH = DATA_DIR / "customers_features.csv"
OUTPUT_PATH = DATA_DIR / "customers_scored.csv"

# -- Score_Mapping sheet, transcribed 1:1 -----------------------------------

OCCUPATION_SCORE_MAP = {
    "科技業": 80, "金融業": 80, "醫療": 80,
    "製造業": 60, "軍公教": 60,
    "服務業": 40, "自由業": 40, "退休": 40,
}

JOB_LEVEL_SCORE_MAP = {"Executive": 100, "Director": 80, "Manager": 60, "Staff": 40}

HAS_FUND_SCORE_MAP = {0: 100, 1: 0}
HAS_ABANDONED_APPLICATION_1M_SCORE_MAP = {0: 0, 1: 100}

# total_aum_score: 0-1,000,000=20 / >1,000,000-5,000,000=40 /
#                  >5,000,000-15,000,000=60 / >15,000,000-30,000,000=80 / >30,000,000=100
TOTAL_AUM_BINS = [-1, 1_000_000, 5_000_000, 15_000_000, 30_000_000, np.inf]
TOTAL_AUM_LABELS = [20, 40, 60, 80, 100]
TOTAL_AUM_RIGHT = True  # (a, b] -- left-open, right-closed

# inflow_1m_score: <=1,000,000=20 / >1,000,000-3,000,000=40 /
#                  >3,000,000-5,000,000=80 / >5,000,000=100
INFLOW_1M_BINS = [-1, 1_000_000, 3_000_000, 5_000_000, np.inf]
INFLOW_1M_LABELS = [20, 40, 80, 100]
INFLOW_1M_RIGHT = True

# fund_view_count_1m_score: 0=20 / 1-10=60 / 11-25=80 / >25=100
FUND_VIEW_COUNT_1M_BINS = [-1, 0, 10, 25, np.inf]
FUND_VIEW_COUNT_1M_LABELS = [20, 60, 80, 100]
FUND_VIEW_COUNT_1M_RIGHT = True

# twd_demand_ratio_score: <10%=20 / 10%-<40%=40 / 40%-<60%=80 / >=60%=100
TWD_DEMAND_RATIO_BINS = [0, 0.10, 0.40, 0.60, np.inf]
TWD_DEMAND_RATIO_LABELS = [20, 40, 80, 100]
TWD_DEMAND_RATIO_RIGHT = False  # [a, b) -- left-closed, right-open

# investment_ratio_score: <10%=100 / 10%-<40%=80 / 40%-<60%=60 / >=60%=40
INVESTMENT_RATIO_BINS = [0, 0.10, 0.40, 0.60, np.inf]
INVESTMENT_RATIO_LABELS = [100, 80, 60, 40]
INVESTMENT_RATIO_RIGHT = False

FEATURE_SCORE_COLUMNS = [
    "occupation_score",
    "job_level_score",
    "has_fund_score",
    "total_aum_score",
    "twd_demand_ratio_score",
    "investment_ratio_score",
    "inflow_1m_score",
    "fund_view_count_1m_score",
    "has_abandoned_application_1m_score",
]

ALLOWED_SCORE_VALUES = {
    "occupation_score": {80, 60, 40},
    "job_level_score": {100, 80, 60, 40},
    "has_fund_score": {100, 0},
    "total_aum_score": {20, 40, 60, 80, 100},
    "twd_demand_ratio_score": {20, 40, 80, 100},
    "investment_ratio_score": {100, 80, 60, 40},
    "inflow_1m_score": {20, 40, 80, 100},
    "fund_view_count_1m_score": {20, 60, 80, 100},
    "has_abandoned_application_1m_score": {0, 100},
}


def load_features_data(path: Path = FEATURES_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def _score_categorical(series: pd.Series, mapping: dict, field_name: str) -> pd.Series:
    result = series.map(mapping)
    unmapped = series[result.isna()]
    if not unmapped.empty:
        bad_values = sorted(unmapped.unique().tolist())
        raise ValueError(f"{field_name} has values not defined in Score_Mapping: {bad_values}")
    return result.astype(int)


def _score_tiered(
    series: pd.Series, bins: list[float], labels: list[int], right: bool, field_name: str
) -> pd.Series:
    if series.isna().any():
        raise ValueError(f"{field_name} has missing (NaN) values that cannot be scored")
    if (series < 0).any():
        bad_values = sorted(series[series < 0].unique().tolist())
        raise ValueError(f"{field_name} has negative values outside the spec's domain: {bad_values}")

    result = pd.cut(series, bins=bins, labels=labels, right=right)
    unmapped = series[result.isna()]
    if not unmapped.empty:
        bad_values = sorted(unmapped.unique().tolist())
        raise ValueError(f"{field_name} has values outside all defined score ranges: {bad_values}")
    return result.astype(int)


def add_feature_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["occupation_score"] = _score_categorical(df["occupation"], OCCUPATION_SCORE_MAP, "occupation")
    df["job_level_score"] = _score_categorical(df["job_level"], JOB_LEVEL_SCORE_MAP, "job_level")
    df["has_fund_score"] = _score_categorical(df["has_fund"], HAS_FUND_SCORE_MAP, "has_fund")

    df["total_aum_score"] = _score_tiered(
        df["total_aum"], TOTAL_AUM_BINS, TOTAL_AUM_LABELS, TOTAL_AUM_RIGHT, "total_aum"
    )
    df["twd_demand_ratio_score"] = _score_tiered(
        df["twd_demand_ratio"], TWD_DEMAND_RATIO_BINS, TWD_DEMAND_RATIO_LABELS,
        TWD_DEMAND_RATIO_RIGHT, "twd_demand_ratio",
    )
    df["investment_ratio_score"] = _score_tiered(
        df["investment_ratio"], INVESTMENT_RATIO_BINS, INVESTMENT_RATIO_LABELS,
        INVESTMENT_RATIO_RIGHT, "investment_ratio",
    )
    df["inflow_1m_score"] = _score_tiered(
        df["inflow_1m"], INFLOW_1M_BINS, INFLOW_1M_LABELS, INFLOW_1M_RIGHT, "inflow_1m"
    )
    df["fund_view_count_1m_score"] = _score_tiered(
        df["fund_view_count_1m"], FUND_VIEW_COUNT_1M_BINS, FUND_VIEW_COUNT_1M_LABELS,
        FUND_VIEW_COUNT_1M_RIGHT, "fund_view_count_1m",
    )
    df["has_abandoned_application_1m_score"] = _score_categorical(
        df["has_abandoned_application_1m"], HAS_ABANDONED_APPLICATION_1M_SCORE_MAP,
        "has_abandoned_application_1m",
    )

    return df


def validate_scores(df_features: pd.DataFrame, df_scored: pd.DataFrame) -> list[str]:
    """Return a list of validation failure messages (empty if all pass)."""
    errors: list[str] = []

    if len(df_scored) != 1000:
        errors.append(f"row count = {len(df_scored)}, expected 1000")

    if len(df_scored) != len(df_features):
        errors.append(f"row count = {len(df_scored)}, expected {len(df_features)} (input rows)")

    if "customer_id" in df_scored.columns and df_scored["customer_id"].duplicated().any():
        dupes = df_scored.loc[df_scored["customer_id"].duplicated(), "customer_id"].tolist()
        errors.append(f"duplicate customer_id values found: {dupes}")

    missing_raw = set(df_features.columns) - set(df_scored.columns)
    if missing_raw:
        errors.append(f"input columns missing from output: {missing_raw}")

    missing_scores = set(FEATURE_SCORE_COLUMNS) - set(df_scored.columns)
    if missing_scores:
        errors.append(f"required score columns missing: {missing_scores}")
        return errors  # remaining checks all depend on these columns existing

    for col in FEATURE_SCORE_COLUMNS:
        if not df_scored[col].between(0, 100).all():
            errors.append(f"{col} has values outside [0, 100]")

        actual_values = set(df_scored[col].unique().tolist())
        unexpected = actual_values - ALLOWED_SCORE_VALUES[col]
        if unexpected:
            errors.append(f"{col} has values not defined in Score_Mapping: {sorted(unexpected)}")

    subscore_or_priority_cols = [
        c for c in df_scored.columns
        if c not in df_features.columns and c not in FEATURE_SCORE_COLUMNS
    ]
    if subscore_or_priority_cols:
        errors.append(f"unexpected extra columns (Sub-Score/Priority belong to Step 5): {subscore_or_priority_cols}")

    return errors


def main() -> None:
    df_features = load_features_data()
    df_scored = add_feature_scores(df_features)

    errors = validate_scores(df_features, df_scored)
    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    df_scored.to_csv(OUTPUT_PATH, index=False)

    print(f"Generated {len(df_scored)} rows -> {OUTPUT_PATH}")
    print("Validation: all checks passed")
    print("\nFeature score summary:")
    print(df_scored[FEATURE_SCORE_COLUMNS].describe().round(2))


if __name__ == "__main__":
    main()
