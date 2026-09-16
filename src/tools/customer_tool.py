"""Phase 2 (WHY), Step 2: Customer data tool.

get_customer_profile(customer_id) retrieves ONE customer's complete 43-field
record from outputs/priority_customers_top10.csv (the WHO Top-N output) and
returns it as a JSON-compatible dict.

This module performs a pure structured-data lookup only:
- no business inference
- no filtering to "WHY-relevant" fields (the LLM selects those later)
- no Priority Score recalculation
- no changes to WHO ranking logic

It is the single entry point Phase 2's insight_generator.py (not implemented
in this step) will call before sending a customer record to the LLM.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TOP10_PATH = Path(__file__).resolve(
).parents[2] / "outputs" / "priority_customers_top10.csv"

# The WHO pipeline's full 43-field output schema (Step5:
# outputs/priority_customers_top10.csv). Kept as an explicit literal here
# (rather than imported from src/ranking) to keep the WHY-side tool
# decoupled from WHO's internal module layout.
EXPECTED_FIELDS = [
    # customer basic info (6)
    "customer_id", "occupation", "job_level", "gender", "age", "risk_level",
    # product holdings (6)
    "has_twd_td", "has_fund", "has_fx_deposit", "has_robo", "has_bond", "has_insurance",
    # asset structure - raw balances (5)
    "twd_demand_balance", "twd_td_balance", "fx_demand_balance", "fx_td_balance", "investment_balance",
    # cashflow (2)
    "inflow_1m", "outflow_1m",
    # fund purchase behavior (3)
    "fund_buy_count_1m", "fund_buy_amount_1m", "has_rm",
    # digital behavior (2)
    "fund_view_count_1m", "has_abandoned_application_1m",
    # derived features (5)
    "total_aum", "twd_demand_ratio", "td_ratio", "fx_ratio", "investment_ratio",
    # feature-level scores (9)
    "occupation_score", "job_level_score", "has_fund_score", "total_aum_score",
    "twd_demand_ratio_score", "investment_ratio_score", "inflow_1m_score",
    "fund_view_count_1m_score", "has_abandoned_application_1m_score",
    # sub-scores + priority (5)
    "asset_potential_score", "investment_gap_score", "investment_intent_score",
    "priority_score", "priority_rank",
]
assert len(EXPECTED_FIELDS) == 43


class CustomerDataError(Exception):
    """Base error for customer_tool failures."""


class CustomerSchemaError(CustomerDataError):
    """Raised when the source CSV's columns don't match EXPECTED_FIELDS."""


class CustomerNotFoundError(CustomerDataError):
    """Raised when the requested customer_id is not in the source data."""


def _load_top10(source_path: Path) -> pd.DataFrame:
    df = pd.read_csv(source_path)

    actual = set(df.columns)
    expected = set(EXPECTED_FIELDS)
    if actual != expected:
        missing = expected - actual
        extra = actual - expected
        raise CustomerSchemaError(
            f"{source_path} does not match the expected 43-field schema. "
            f"missing={sorted(missing)}, unexpected={sorted(extra)}"
        )
    return df


def _row_to_json_compatible_dict(row: pd.Series) -> dict[str, Any]:
    """Convert a pandas row to a plain dict of native str/int/float values
    (no numpy scalar types), preserving the 43-field column order."""
    result: dict[str, Any] = {}
    for field in EXPECTED_FIELDS:
        value = row[field]
        if isinstance(value, np.generic):
            value = value.item()
        result[field] = value
    return result


def get_customer_profile(customer_id: str, source_path: Path = TOP10_PATH) -> dict[str, Any]:
    """Return the complete 43-field record for one customer as a
    JSON-compatible dict.

    Raises:
        CustomerSchemaError: if source_path's columns don't match the
            expected 43-field WHO output schema.
        CustomerNotFoundError: if customer_id is not present in source_path.
        CustomerDataError: if more than one row matches customer_id (data
            integrity issue -- customer_id should be unique in the WHO output).
    """
    df = _load_top10(source_path)

    matches = df[df["customer_id"] == customer_id]

    if matches.empty:
        raise CustomerNotFoundError(
            f"customer_id '{customer_id}' was not found in {source_path}"
        )
    if len(matches) > 1:
        raise CustomerDataError(
            f"customer_id '{customer_id}' matched {len(matches)} rows in {source_path}, "
            "expected exactly 1 (customer_id should be unique)"
        )

    return _row_to_json_compatible_dict(matches.iloc[0])


def main() -> None:
    """Manual demo: print one customer's profile as JSON.

    Usage:
        python src/tools/customer_tool.py             # rank-1 customer
        python src/tools/customer_tool.py C000104      # a specific customer_id
    """
    if len(sys.argv) > 1:
        customer_id = sys.argv[1]
    else:
        df = _load_top10(TOP10_PATH)
        customer_id = df.loc[df["priority_rank"].idxmin(), "customer_id"]

    profile = get_customer_profile(customer_id)
    print(json.dumps(profile, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
