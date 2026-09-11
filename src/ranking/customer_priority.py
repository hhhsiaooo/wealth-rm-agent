"""Step 5: Sub-Scores, Priority Score, and Ranking.

Reads data/synthetic/customers_scored.csv (feature-level scores from Step 4
are NOT recalculated), preserves every existing column, and appends:

    asset_potential_score, investment_gap_score, investment_intent_score,
    priority_score, priority_rank

per docs/ai_product_data_definition.xlsx (SubScore_Mapping and
PriorityScore_Mapping sheets).

Ranking rule:
- Sort by priority_score descending, customer_id ascending as the
  deterministic tie-break (so priority_rank is a unique, reproducible
  sequence from 1 to N regardless of input row order).
- Top 10 selection uses a SCORE THRESHOLD, not "rank <= 10": the cutoff is
  the priority_score of the 10th-ranked customer, and every customer whose
  priority_score is >= that cutoff is included. This means the Top 10 file
  can contain more than 10 rows when there is a tie at the boundary
  (confirmed business rule), while priority_rank itself stays a strict,
  unique 1..N ordering.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "synthetic"
OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"

SCORED_PATH = DATA_DIR / "customers_scored.csv"
ALL_OUTPUT_PATH = OUTPUTS_DIR / "priority_customers_all.csv"
TOP10_OUTPUT_PATH = OUTPUTS_DIR / "priority_customers_top10.csv"

TOP_N = 10

# -- SubScore_Mapping sheet, transcribed 1:1 --------------------------------
SUBSCORE_WEIGHTS = {
    "asset_potential_score": {
        "total_aum_score": 0.70,
        "occupation_score": 0.15,
        "job_level_score": 0.15,
    },
    "investment_gap_score": {
        "twd_demand_ratio_score": 0.40,
        "total_aum_score": 0.30,
        "investment_ratio_score": 0.20,
        "has_fund_score": 0.10,
    },
    "investment_intent_score": {
        "inflow_1m_score": 0.40,
        "has_abandoned_application_1m_score": 0.35,
        "fund_view_count_1m_score": 0.25,
    },
}

SUBSCORE_COLUMNS = ["asset_potential_score", "investment_gap_score", "investment_intent_score"]

# -- PriorityScore_Mapping sheet, transcribed 1:1 ---------------------------
PRIORITY_WEIGHTS = {
    "investment_gap_score": 0.45,
    "investment_intent_score": 0.35,
    "asset_potential_score": 0.20,
}

FEATURE_SCORE_COLUMNS = [
    "occupation_score", "job_level_score", "has_fund_score", "total_aum_score",
    "twd_demand_ratio_score", "investment_ratio_score", "inflow_1m_score",
    "fund_view_count_1m_score", "has_abandoned_application_1m_score",
]


def load_scored_data(path: Path = SCORED_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def compute_subscores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for subscore_col, weights in SUBSCORE_WEIGHTS.items():
        df[subscore_col] = sum(df[feature_col] * weight for feature_col, weight in weights.items())
    return df


def compute_priority_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["priority_score"] = sum(df[col] * weight for col, weight in PRIORITY_WEIGHTS.items())
    return df


def assign_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by priority_score descending, customer_id ascending (deterministic
    tie-break), and assign a unique priority_rank from 1 to len(df)."""
    df = df.sort_values(
        by=["priority_score", "customer_id"], ascending=[False, True]
    ).reset_index(drop=True)
    df["priority_rank"] = df.index + 1
    return df


def select_top_n_with_ties(df_ranked: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    """Return the top n rows of df_ranked (already sorted descending), but
    extend the cutoff to include any customer tied with the nth-place
    priority_score, so ties at the boundary are never silently dropped."""
    if len(df_ranked) <= n:
        return df_ranked.copy()
    cutoff_score = df_ranked.iloc[n - 1]["priority_score"]
    return df_ranked[df_ranked["priority_score"] >= cutoff_score].copy()


def build_priority_ranking(df_scored: pd.DataFrame) -> pd.DataFrame:
    df = compute_subscores(df_scored)
    df = compute_priority_score(df)
    df = assign_rank(df)
    return df


def validate_priority(df_scored: pd.DataFrame, df_ranked: pd.DataFrame) -> list[str]:
    """Return a list of validation failure messages (empty if all pass)."""
    errors: list[str] = []

    if len(df_ranked) != len(df_scored):
        errors.append(f"row count = {len(df_ranked)}, expected {len(df_scored)}")

    required_cols = SUBSCORE_COLUMNS + ["priority_score", "priority_rank"]
    missing = set(required_cols) - set(df_ranked.columns)
    if missing:
        errors.append(f"required columns missing: {missing}")
        return errors

    for col in SUBSCORE_COLUMNS + ["priority_score"]:
        if not df_ranked[col].between(0, 100).all():
            errors.append(f"{col} has values outside [0, 100]")

    # Feature scores must not have been recalculated / mutated.
    left = df_scored.set_index("customer_id")[FEATURE_SCORE_COLUMNS].sort_index()
    right = df_ranked.set_index("customer_id")[FEATURE_SCORE_COLUMNS].sort_index()
    if not left.equals(right):
        errors.append("feature-level score columns were altered from customers_scored.csv")

    ranks = df_ranked["priority_rank"]
    if not ranks.is_unique:
        errors.append("priority_rank has duplicate values")
    if sorted(ranks.tolist()) != list(range(1, len(df_ranked) + 1)):
        errors.append(f"priority_rank does not cover a contiguous 1..{len(df_ranked)} sequence")

    ordered = df_ranked.sort_values("priority_rank")
    if not ordered["priority_score"].is_monotonic_decreasing:
        errors.append("priority_score is not sorted in descending order by priority_rank")

    if df_ranked["customer_id"].duplicated().any():
        errors.append("duplicate customer_id values found")

    return errors


def validate_top10(df_ranked: pd.DataFrame, df_top10: pd.DataFrame) -> list[str]:
    errors: list[str] = []

    if len(df_top10) < TOP_N:
        errors.append(f"top10 output has fewer than {TOP_N} rows: {len(df_top10)}")

    if not set(df_top10["customer_id"]).issubset(set(df_ranked["customer_id"])):
        errors.append("top10 output contains customer_id values not present in the full ranking")

    merged = df_top10.merge(df_ranked, on=list(df_top10.columns), how="left", indicator=True)
    if (merged["_merge"] != "both").any():
        errors.append("top10 rows are not an exact subset of the full ranking (value mismatch)")

    if len(df_ranked) > TOP_N:
        cutoff_score = df_ranked.sort_values("priority_rank").iloc[TOP_N - 1]["priority_score"]
        expected_ids = set(df_ranked.loc[df_ranked["priority_score"] >= cutoff_score, "customer_id"])
        if set(df_top10["customer_id"]) != expected_ids:
            errors.append("top10 does not exactly match the score-threshold cutoff at rank 10")

    return errors


def main() -> None:
    df_scored = load_scored_data()
    df_ranked = build_priority_ranking(df_scored)

    errors = validate_priority(df_scored, df_ranked)
    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    df_top10 = select_top_n_with_ties(df_ranked, TOP_N)

    top10_errors = validate_top10(df_ranked, df_top10)
    if top10_errors:
        print("TOP10 VALIDATION FAILED:")
        for e in top10_errors:
            print(f"  - {e}")
        raise SystemExit(1)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df_ranked.to_csv(ALL_OUTPUT_PATH, index=False)
    df_top10.to_csv(TOP10_OUTPUT_PATH, index=False)

    print(f"Generated {len(df_ranked)} rows -> {ALL_OUTPUT_PATH}")
    print(f"Generated {len(df_top10)} rows -> {TOP10_OUTPUT_PATH}")
    print("Validation: all checks passed")


if __name__ == "__main__":
    main()
