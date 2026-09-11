import pandas as pd
import pytest

from ranking.customer_scoring import (
    FEATURE_SCORE_COLUMNS,
    FEATURES_PATH,
    add_feature_scores,
    load_features_data,
    validate_scores,
)

REQUIRED_INPUT_COLUMNS = [
    "customer_id", "occupation", "job_level", "has_fund",
    "total_aum", "twd_demand_ratio", "investment_ratio",
    "inflow_1m", "fund_view_count_1m", "has_abandoned_application_1m",
]


def make_features_df(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "customer_id": None,
        "occupation": "科技業",
        "job_level": "Staff",
        "has_fund": 0,
        "total_aum": 0,
        "twd_demand_ratio": 0.0,
        "investment_ratio": 0.0,
        "inflow_1m": 0,
        "fund_view_count_1m": 0,
        "has_abandoned_application_1m": 0,
    }
    records = []
    for i, row in enumerate(rows):
        record = dict(defaults)
        record.update(row)
        record["customer_id"] = record["customer_id"] or f"C{i:06d}"
        records.append(record)
    return pd.DataFrame(records, columns=REQUIRED_INPUT_COLUMNS)


# -- occupation_score / job_level_score (categorical) ------------------------

@pytest.mark.parametrize("occupation,expected", [
    ("科技業", 80), ("金融業", 80), ("醫療", 80),
    ("製造業", 60), ("軍公教", 60),
    ("服務業", 40), ("自由業", 40), ("退休", 40),
])
def test_occupation_score_every_category(occupation, expected):
    df = make_features_df([{"occupation": occupation}])
    result = add_feature_scores(df)
    assert result.loc[0, "occupation_score"] == expected


def test_occupation_score_unknown_category_raises():
    df = make_features_df([{"occupation": "外星業"}])
    with pytest.raises(ValueError, match="occupation"):
        add_feature_scores(df)


@pytest.mark.parametrize("job_level,expected", [
    ("Executive", 100), ("Director", 80), ("Manager", 60), ("Staff", 40),
])
def test_job_level_score_every_category(job_level, expected):
    df = make_features_df([{"job_level": job_level}])
    result = add_feature_scores(df)
    assert result.loc[0, "job_level_score"] == expected


def test_job_level_score_unknown_category_raises():
    df = make_features_df([{"job_level": "Intern"}])
    with pytest.raises(ValueError, match="job_level"):
        add_feature_scores(df)


# -- has_fund_score / has_abandoned_application_1m_score (boolean) -----------

@pytest.mark.parametrize("has_fund,expected", [(0, 100), (1, 0)])
def test_has_fund_score(has_fund, expected):
    df = make_features_df([{"has_fund": has_fund}])
    result = add_feature_scores(df)
    assert result.loc[0, "has_fund_score"] == expected


def test_has_fund_score_invalid_value_raises():
    df = make_features_df([{"has_fund": 2}])
    with pytest.raises(ValueError, match="has_fund"):
        add_feature_scores(df)


@pytest.mark.parametrize("has_abandoned,expected", [(0, 0), (1, 100)])
def test_has_abandoned_application_1m_score(has_abandoned, expected):
    df = make_features_df([{"has_abandoned_application_1m": has_abandoned}])
    result = add_feature_scores(df)
    assert result.loc[0, "has_abandoned_application_1m_score"] == expected


def test_has_abandoned_application_1m_score_invalid_value_raises():
    df = make_features_df([{"has_abandoned_application_1m": 5}])
    with pytest.raises(ValueError, match="has_abandoned_application_1m"):
        add_feature_scores(df)


# -- total_aum_score: 0-1M=20 / >1M-5M=40 / >5M-15M=60 / >15M-30M=80 / >30M=100

@pytest.mark.parametrize("total_aum,expected", [
    (0, 20),
    (1_000_000, 20),          # boundary: right-closed, stays in lower tier
    (1_000_001, 40),          # boundary: just above, moves to next tier
    (5_000_000, 40),
    (5_000_001, 60),
    (15_000_000, 60),
    (15_000_001, 80),
    (30_000_000, 80),
    (30_000_001, 100),
    (100_000_000, 100),       # far above top boundary
])
def test_total_aum_score_boundaries(total_aum, expected):
    df = make_features_df([{"total_aum": total_aum}])
    result = add_feature_scores(df)
    assert result.loc[0, "total_aum_score"] == expected


def test_total_aum_score_negative_raises():
    df = make_features_df([{"total_aum": -1}])
    with pytest.raises(ValueError, match="total_aum"):
        add_feature_scores(df)


# -- twd_demand_ratio_score: <10%=20 / 10%-<40%=40 / 40%-<60%=80 / >=60%=100

@pytest.mark.parametrize("ratio,expected", [
    (0.0, 20),
    (0.0999, 20),
    (0.10, 40),                # boundary: left-closed, moves up at exactly 10%
    (0.3999, 40),
    (0.40, 80),
    (0.5999, 80),
    (0.60, 100),
    (1.0, 100),
])
def test_twd_demand_ratio_score_boundaries(ratio, expected):
    df = make_features_df([{"twd_demand_ratio": ratio}])
    result = add_feature_scores(df)
    assert result.loc[0, "twd_demand_ratio_score"] == expected


# -- investment_ratio_score: <10%=100 / 10%-<40%=80 / 40%-<60%=60 / >=60%=40 (inverse direction)

@pytest.mark.parametrize("ratio,expected", [
    (0.0, 100),
    (0.0999, 100),
    (0.10, 80),
    (0.3999, 80),
    (0.40, 60),
    (0.5999, 60),
    (0.60, 40),
    (1.0, 40),
])
def test_investment_ratio_score_boundaries(ratio, expected):
    df = make_features_df([{"investment_ratio": ratio}])
    result = add_feature_scores(df)
    assert result.loc[0, "investment_ratio_score"] == expected


# -- inflow_1m_score: <=1M=20 / >1M-3M=40 / >3M-5M=80 / >5M=100

@pytest.mark.parametrize("inflow,expected", [
    (0, 20),
    (1_000_000, 20),
    (1_000_001, 40),
    (3_000_000, 40),
    (3_000_001, 80),
    (5_000_000, 80),
    (5_000_001, 100),
])
def test_inflow_1m_score_boundaries(inflow, expected):
    df = make_features_df([{"inflow_1m": inflow}])
    result = add_feature_scores(df)
    assert result.loc[0, "inflow_1m_score"] == expected


# -- fund_view_count_1m_score: 0=20 / 1-10=60 / 11-25=80 / >25=100

@pytest.mark.parametrize("count,expected", [
    (0, 20),
    (1, 60),
    (10, 60),
    (11, 80),
    (25, 80),
    (26, 100),
    (30, 100),
])
def test_fund_view_count_1m_score_boundaries(count, expected):
    df = make_features_df([{"fund_view_count_1m": count}])
    result = add_feature_scores(df)
    assert result.loc[0, "fund_view_count_1m_score"] == expected


# -- generic properties --------------------------------------------------

def test_all_score_columns_present_and_in_range():
    df = make_features_df([{"occupation": "退休", "job_level": "Executive"}])
    result = add_feature_scores(df)
    for col in FEATURE_SCORE_COLUMNS:
        assert col in result.columns
        assert 0 <= result.loc[0, col] <= 100


def test_input_columns_are_preserved():
    df = make_features_df([{"occupation": "醫療"}])
    result = add_feature_scores(df)
    for col in df.columns:
        assert col in result.columns


def test_no_subscore_or_priority_columns_added():
    df = make_features_df([{"occupation": "醫療"}])
    result = add_feature_scores(df)
    forbidden = ["priority", "sub_score", "subscore", "rank"]
    for col in result.columns:
        for token in forbidden:
            assert token not in col.lower()


# -- validate_scores -------------------------------------------------------

def test_validate_scores_passes_on_correct_output_1000_rows():
    rows = [{"occupation": "科技業", "total_aum": i * 40_000} for i in range(1000)]
    df = make_features_df(rows)
    result = add_feature_scores(df)
    errors = validate_scores(df, result)
    assert errors == []


def test_validate_scores_detects_duplicate_customer_id():
    df = make_features_df([{"customer_id": "C000001"}, {"customer_id": "C000001"}])
    result = add_feature_scores(df)
    errors = validate_scores(df, result)
    assert any("duplicate" in e for e in errors)


def test_validate_scores_detects_missing_score_column():
    df = make_features_df([{}])
    result = add_feature_scores(df).drop(columns=["total_aum_score"])
    errors = validate_scores(df, result)
    assert any("total_aum_score" in e for e in errors)


def test_validate_scores_detects_out_of_range_value():
    df = make_features_df([{}])
    result = add_feature_scores(df)
    result.loc[0, "occupation_score"] = 150
    errors = validate_scores(df, result)
    assert any("occupation_score" in e for e in errors)


@pytest.mark.skipif(not FEATURES_PATH.exists(), reason="customers_features.csv not generated yet")
def test_end_to_end_on_actual_features_data():
    df_features = load_features_data()
    df_scored = add_feature_scores(df_features)

    errors = validate_scores(df_features, df_scored)
    assert errors == []
    assert len(df_scored) == 1000
    for col in FEATURE_SCORE_COLUMNS:
        assert df_scored[col].between(0, 100).all()
