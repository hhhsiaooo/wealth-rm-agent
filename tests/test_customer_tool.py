from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tools.customer_tool import (
    EXPECTED_FIELDS,
    TOP10_PATH,
    CustomerDataError,
    CustomerNotFoundError,
    CustomerSchemaError,
    get_customer_profile,
)


def make_row(customer_id: str, priority_rank: int = 1) -> dict:
    """A complete, valid 43-field row with sensible default values."""
    row = {field: 0 for field in EXPECTED_FIELDS}
    row.update({
        "customer_id": customer_id,
        "occupation": "科技業",
        "job_level": "Manager",
        "gender": "Female",
        "age": 40,
        "risk_level": "Medium",
        "priority_rank": priority_rank,
        "priority_score": 90.5,
    })
    return row


def write_csv(tmp_path, rows: list[dict], columns: list[str] | None = None) -> Path:
    df = pd.DataFrame(rows, columns=columns or EXPECTED_FIELDS)
    path = tmp_path / "priority_customers_top10.csv"
    df.to_csv(path, index=False)
    return path


# -- happy path ---------------------------------------------------------

def test_returns_all_43_fields(tmp_path):
    path = write_csv(tmp_path, [make_row("C000001")])
    profile = get_customer_profile("C000001", source_path=path)
    assert set(profile.keys()) == set(EXPECTED_FIELDS)
    assert len(profile) == 43


def test_returns_the_correct_customer(tmp_path):
    path = write_csv(tmp_path, [make_row("C000001", 1), make_row("C000002", 2)])
    profile = get_customer_profile("C000002", source_path=path)
    assert profile["customer_id"] == "C000002"
    assert profile["priority_rank"] == 2


def test_values_match_source_row(tmp_path):
    path = write_csv(tmp_path, [make_row("C000001")])
    profile = get_customer_profile("C000001", source_path=path)

    df = pd.read_csv(path)
    source_row = df.iloc[0]
    for field in EXPECTED_FIELDS:
        assert profile[field] == source_row[field]


def test_does_not_recalculate_priority_score(tmp_path):
    row = make_row("C000001")
    row["priority_score"] = 77.7
    path = write_csv(tmp_path, [row])
    profile = get_customer_profile("C000001", source_path=path)
    assert profile["priority_score"] == 77.7


def test_output_is_json_serializable_with_native_types(tmp_path):
    path = write_csv(tmp_path, [make_row("C000001")])
    profile = get_customer_profile("C000001", source_path=path)

    serialized = json.dumps(profile, ensure_ascii=False)
    assert json.loads(serialized) == profile

    for value in profile.values():
        assert type(value) in (str, int, float)


def test_boolean_like_fields_stay_as_int_not_python_bool(tmp_path):
    row = make_row("C000001")
    row["has_fund"] = 1
    row["has_abandoned_application_1m"] = 0
    path = write_csv(tmp_path, [row])
    profile = get_customer_profile("C000001", source_path=path)

    assert profile["has_fund"] == 1 and type(profile["has_fund"]) is int
    assert profile["has_abandoned_application_1m"] == 0 and type(profile["has_abandoned_application_1m"]) is int


def test_field_order_matches_expected_fields(tmp_path):
    path = write_csv(tmp_path, [make_row("C000001")])
    profile = get_customer_profile("C000001", source_path=path)
    assert list(profile.keys()) == EXPECTED_FIELDS


# -- error handling -------------------------------------------------------

def test_unknown_customer_id_raises_not_found(tmp_path):
    path = write_csv(tmp_path, [make_row("C000001")])
    with pytest.raises(CustomerNotFoundError, match="C999999"):
        get_customer_profile("C999999", source_path=path)


def test_not_found_error_is_a_customer_data_error(tmp_path):
    path = write_csv(tmp_path, [make_row("C000001")])
    with pytest.raises(CustomerDataError):
        get_customer_profile("C999999", source_path=path)


def test_missing_column_raises_schema_error(tmp_path):
    columns = [f for f in EXPECTED_FIELDS if f != "fx_ratio"]
    path = write_csv(tmp_path, [
        {k: v for k, v in make_row("C000001").items() if k != "fx_ratio"}
    ], columns=columns)
    with pytest.raises(CustomerSchemaError, match="fx_ratio"):
        get_customer_profile("C000001", source_path=path)


def test_extra_column_raises_schema_error(tmp_path):
    row = make_row("C000001")
    row["unexpected_extra_field"] = "surprise"
    columns = EXPECTED_FIELDS + ["unexpected_extra_field"]
    path = write_csv(tmp_path, [row], columns=columns)
    with pytest.raises(CustomerSchemaError, match="unexpected_extra_field"):
        get_customer_profile("C000001", source_path=path)


def test_duplicate_customer_id_raises_error(tmp_path):
    path = write_csv(tmp_path, [make_row("C000001", 1), make_row("C000001", 2)])
    with pytest.raises(CustomerDataError, match="C000001"):
        get_customer_profile("C000001", source_path=path)


# -- end-to-end on actual WHO output --------------------------------------

@pytest.mark.skipif(not TOP10_PATH.exists(), reason="priority_customers_top10.csv not generated yet")
def test_end_to_end_on_actual_top10_file():
    df = pd.read_csv(TOP10_PATH)
    sample_id = df.iloc[0]["customer_id"]

    profile = get_customer_profile(sample_id)

    assert profile["customer_id"] == sample_id
    assert len(profile) == 43
    json.dumps(profile)  # must not raise


@pytest.mark.skipif(not TOP10_PATH.exists(), reason="priority_customers_top10.csv not generated yet")
def test_every_customer_in_actual_top10_file_is_retrievable():
    df = pd.read_csv(TOP10_PATH)
    for customer_id in df["customer_id"]:
        profile = get_customer_profile(customer_id)
        assert profile["customer_id"] == customer_id
