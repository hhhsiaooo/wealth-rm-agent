import pandas as pd
import pytest

from ranking.customer_priority import (
    FEATURE_SCORE_COLUMNS,
    SCORED_PATH,
    TOP_N,
    assign_rank,
    build_priority_ranking,
    compute_priority_score,
    compute_subscores,
    load_scored_data,
    select_top_n_with_ties,
    validate_priority,
    validate_top10,
)


def make_scored_df(rows: list[dict]) -> pd.DataFrame:
    defaults = {col: 0 for col in FEATURE_SCORE_COLUMNS}
    records = []
    for i, row in enumerate(rows):
        record = {"customer_id": row.get("customer_id", f"C{i:06d}")}
        record.update(defaults)
        record.update({k: v for k, v in row.items() if k != "customer_id"})
        records.append(record)
    return pd.DataFrame(records)


# -- SubScore formulas -------------------------------------------------------

def test_asset_potential_score_formula():
    df = make_scored_df([{
        "total_aum_score": 60, "occupation_score": 80, "job_level_score": 40,
    }])
    result = compute_subscores(df)
    expected = 60 * 0.70 + 80 * 0.15 + 40 * 0.15
    assert result.loc[0, "asset_potential_score"] == pytest.approx(expected)


def test_investment_gap_score_formula():
    df = make_scored_df([{
        "twd_demand_ratio_score": 100, "total_aum_score": 60,
        "investment_ratio_score": 40, "has_fund_score": 0,
    }])
    result = compute_subscores(df)
    expected = 100 * 0.40 + 60 * 0.30 + 40 * 0.20 + 0 * 0.10
    assert result.loc[0, "investment_gap_score"] == pytest.approx(expected)


def test_investment_intent_score_formula():
    df = make_scored_df([{
        "inflow_1m_score": 100, "has_abandoned_application_1m_score": 100,
        "fund_view_count_1m_score": 80,
    }])
    result = compute_subscores(df)
    expected = 100 * 0.40 + 100 * 0.35 + 80 * 0.25
    assert result.loc[0, "investment_intent_score"] == pytest.approx(expected)


# -- Priority Score formula ---------------------------------------------------

def test_priority_score_formula():
    df = make_scored_df([{}])
    df = compute_subscores(df)
    df.loc[0, "investment_gap_score"] = 80
    df.loc[0, "investment_intent_score"] = 60
    df.loc[0, "asset_potential_score"] = 40
    result = compute_priority_score(df)
    expected = 80 * 0.45 + 60 * 0.35 + 40 * 0.20
    assert result.loc[0, "priority_score"] == pytest.approx(expected)


# -- score range ---------------------------------------------------------

def test_all_zero_feature_scores_give_zero_subscores_and_priority():
    df = make_scored_df([{}])  # all feature scores default to 0
    df = compute_subscores(df)
    df = compute_priority_score(df)
    for col in ["asset_potential_score", "investment_gap_score", "investment_intent_score", "priority_score"]:
        assert df.loc[0, col] == 0


def test_all_max_feature_scores_give_max_subscores_and_priority():
    df = make_scored_df([{col: 100 for col in FEATURE_SCORE_COLUMNS}])
    df = compute_subscores(df)
    df = compute_priority_score(df)
    for col in ["asset_potential_score", "investment_gap_score", "investment_intent_score", "priority_score"]:
        assert df.loc[0, col] == pytest.approx(100)


def test_subscores_and_priority_score_always_within_0_100():
    import itertools
    rows = []
    for combo in itertools.product([0, 100], repeat=3):
        row = {c: 0 for c in FEATURE_SCORE_COLUMNS}
        row["total_aum_score"], row["occupation_score"], row["job_level_score"] = [
            v for v in combo
        ]
        rows.append(row)
    df = make_scored_df(rows)
    df = compute_subscores(df)
    df = compute_priority_score(df)
    for col in ["asset_potential_score", "investment_gap_score", "investment_intent_score", "priority_score"]:
        assert df[col].between(0, 100).all()


# -- rank uniqueness / descending order --------------------------------------

def test_priority_rank_is_unique_1_to_n():
    df = make_scored_df([
        {"customer_id": f"C{i:06d}", "total_aum_score": (i * 7) % 101}
        for i in range(50)
    ])
    df = compute_subscores(df)
    df = compute_priority_score(df)
    df = assign_rank(df)

    assert sorted(df["priority_rank"].tolist()) == list(range(1, 51))


def test_priority_rank_descending_order():
    df = make_scored_df([
        {"customer_id": f"C{i:06d}", "total_aum_score": (i * 13) % 101}
        for i in range(30)
    ])
    df = compute_subscores(df)
    df = compute_priority_score(df)
    df = assign_rank(df)

    ordered = df.sort_values("priority_rank")
    assert ordered["priority_score"].is_monotonic_decreasing


def test_tie_break_uses_customer_id_deterministically():
    # Two customers with identical feature scores (identical priority_score).
    df = make_scored_df([
        {"customer_id": "C000002", "total_aum_score": 60},
        {"customer_id": "C000001", "total_aum_score": 60},
    ])
    df = compute_subscores(df)
    df = compute_priority_score(df)
    df = assign_rank(df)

    ordered = df.sort_values("priority_rank")
    assert ordered["customer_id"].tolist() == ["C000001", "C000002"]


# -- Top 10 correctness -------------------------------------------------------

def test_top10_is_exact_subset_of_full_ranking_no_tie():
    df = make_scored_df([
        {"customer_id": f"C{i:06d}", "total_aum_score": i}  # strictly increasing -> no ties
        for i in range(30)
    ])
    df = compute_subscores(df)
    df = compute_priority_score(df)
    df_ranked = assign_rank(df)

    df_top10 = select_top_n_with_ties(df_ranked, TOP_N)
    assert len(df_top10) == TOP_N

    merged = df_top10.merge(df_ranked, on=list(df_top10.columns), how="left", indicator=True)
    assert (merged["_merge"] == "both").all()


def test_top10_extends_past_10_on_boundary_tie():
    rows = [{"customer_id": f"C{i:06d}", "total_aum_score": 100 - i} for i in range(9)]  # ranks 1-9, distinct
    # Ranks 10 and 11 tie at the same score.
    rows.append({"customer_id": "C000010", "total_aum_score": 50})
    rows.append({"customer_id": "C000011", "total_aum_score": 50})
    # A clearly lower-scored customer that must NOT be included.
    rows.append({"customer_id": "C000012", "total_aum_score": 0})

    df = make_scored_df(rows)
    df = compute_subscores(df)
    df = compute_priority_score(df)
    df_ranked = assign_rank(df)

    df_top10 = select_top_n_with_ties(df_ranked, TOP_N)

    assert len(df_top10) == 11
    assert {"C000010", "C000011"}.issubset(set(df_top10["customer_id"]))
    assert "C000012" not in set(df_top10["customer_id"])


def test_top10_no_extension_when_no_tie_at_boundary():
    df = make_scored_df([
        {"customer_id": f"C{i:06d}", "total_aum_score": 100 - i} for i in range(20)
    ])
    df = compute_subscores(df)
    df = compute_priority_score(df)
    df_ranked = assign_rank(df)

    df_top10 = select_top_n_with_ties(df_ranked, TOP_N)
    assert len(df_top10) == TOP_N


def test_validate_top10_passes_on_boundary_tie_case():
    rows = [{"customer_id": f"C{i:06d}", "total_aum_score": 100 - i} for i in range(9)]
    rows.append({"customer_id": "C000010", "total_aum_score": 50})
    rows.append({"customer_id": "C000011", "total_aum_score": 50})
    df = make_scored_df(rows)
    df = compute_subscores(df)
    df = compute_priority_score(df)
    df_ranked = assign_rank(df)
    df_top10 = select_top_n_with_ties(df_ranked, TOP_N)

    assert validate_top10(df_ranked, df_top10) == []


# -- reproducibility ----------------------------------------------------------

def test_ranking_is_reproducible_regardless_of_input_row_order():
    rows = [
        {"customer_id": f"C{i:06d}", "total_aum_score": (i * 31) % 101}
        for i in range(40)
    ]
    df_a = make_scored_df(rows)
    df_b = make_scored_df(list(reversed(rows)))

    ranked_a = build_priority_ranking(df_a).set_index("customer_id").sort_index()
    ranked_b = build_priority_ranking(df_b).set_index("customer_id").sort_index()

    pd.testing.assert_series_equal(ranked_a["priority_rank"], ranked_b["priority_rank"])
    pd.testing.assert_series_equal(ranked_a["priority_score"], ranked_b["priority_score"])


def test_build_priority_ranking_called_twice_gives_identical_result():
    rows = [{"customer_id": f"C{i:06d}", "total_aum_score": i * 3 % 101} for i in range(25)]
    df = make_scored_df(rows)

    result1 = build_priority_ranking(df)
    result2 = build_priority_ranking(df)

    pd.testing.assert_frame_equal(result1, result2)


# -- validate_priority does not allow feature-score mutation ------------------

def test_validate_priority_detects_mutated_feature_score():
    df = make_scored_df([{"total_aum_score": 60}])
    ranked = build_priority_ranking(df)
    ranked.loc[0, "total_aum_score"] = 20  # simulate accidental recalculation
    errors = validate_priority(df, ranked)
    assert any("feature-level score" in e for e in errors)


def test_validate_priority_passes_on_correct_output():
    rows = [{"customer_id": f"C{i:06d}", "total_aum_score": (i * 17) % 101} for i in range(100)]
    df = make_scored_df(rows)
    ranked = build_priority_ranking(df)
    assert validate_priority(df, ranked) == []


# -- end-to-end on actual data -------------------------------------------------

@pytest.mark.skipif(not SCORED_PATH.exists(), reason="customers_scored.csv not generated yet")
def test_end_to_end_on_actual_scored_data():
    df_scored = load_scored_data()
    df_ranked = build_priority_ranking(df_scored)

    assert validate_priority(df_scored, df_ranked) == []
    assert len(df_ranked) == 1000

    df_top10 = select_top_n_with_ties(df_ranked, TOP_N)
    assert validate_top10(df_ranked, df_top10) == []
    assert len(df_top10) >= TOP_N
