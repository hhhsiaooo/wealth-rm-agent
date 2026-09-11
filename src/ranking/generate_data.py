"""Step 2: Synthetic raw customer data generation.

Generates raw customer fields only, per docs/ai_product_data_definition.xlsx
(Data_Specification sheet). Derived features (total_aum, *_ratio) are NOT
generated here -- total_aum is computed internally only as a generation aid
(to allocate correlated balances and cap inflow/outflow) and is discarded
before the output is written.

Customer archetypes (docs/ai_product_data_definition.xlsx, Customer_Archetype
sheet) are used only to bias the random sampling of correlated raw fields so
part of the population contains realistic, explainable customer shapes.
Of the 1,000 customers, 600 are archetype-guided (120 per archetype x 5
archetypes) and 400 are "Unconstrained" -- generated from the general
Data_Specification distributions with no archetype targeting, since real
customers are not limited to the 5 defined shapes. Archetype labels are
never written to the output and never used to compute any score.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_CUSTOMERS = 1000

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "synthetic"
OUTPUT_PATH = RAW_DIR / "customers_raw.csv"

RAW_COLUMNS = [
    "customer_id",
    "occupation",
    "job_level",
    "gender",
    "age",
    "risk_level",
    "has_twd_td",
    "has_fund",
    "has_fx_deposit",
    "has_robo",
    "has_bond",
    "has_insurance",
    "twd_demand_balance",
    "twd_td_balance",
    "fx_demand_balance",
    "fx_td_balance",
    "investment_balance",
    "inflow_1m",
    "outflow_1m",
    "fund_buy_count_1m",
    "fund_buy_amount_1m",
    "has_rm",
    "fund_view_count_1m",
    "has_abandoned_application_1m",
]

# -- General (non-archetype) categorical distributions --------------------
# Converted from the qualitative hints in Data_Specification.GenerationRule
# into concrete probabilities. These are generation-design assumptions, not
# values stated verbatim in the spec.
OCCUPATION_CATEGORIES = ["科技業", "金融業", "醫療", "製造業", "軍公教", "服務業", "自由業", "退休"]
OCCUPATION_WEIGHTS = [0.18, 0.18, 0.14, 0.12, 0.12, 0.10, 0.08, 0.08]

JOB_LEVEL_CATEGORIES = ["Staff", "Manager", "Director", "Executive"]
JOB_LEVEL_WEIGHTS = [0.35, 0.30, 0.25, 0.10]

GENDER_CATEGORIES = ["Male", "Female"]
GENDER_WEIGHTS = [0.5, 0.5]

RISK_LEVEL_CATEGORIES = ["Low", "Medium", "High"]
RISK_LEVEL_WEIGHTS = [0.40, 0.40, 0.20]

# All monetary fields are whole TWD amounts (no cents in this spec).
AMOUNT_COLUMNS = [
    "twd_demand_balance",
    "twd_td_balance",
    "fx_demand_balance",
    "fx_td_balance",
    "investment_balance",
    "inflow_1m",
    "outflow_1m",
    "fund_buy_amount_1m",
]

# Field ranges (spec upper bounds)
TWD_DEMAND_MAX = 12_000_000
TWD_TD_MAX = 8_000_000
FX_DEMAND_MAX = 6_000_000
FX_TD_MAX = 6_000_000
INVESTMENT_MAX = 10_000_000
INFLOW_MAX = 10_000_000
OUTFLOW_MAX = 5_000_000
CASHFLOW_TO_AUM_CAP = 0.2  # inflow_1m / outflow_1m must not exceed 0.2 x total_aum

# -- Archetype generation guidance -----------------------------------------
# Numeric ranges reuse the tier boundaries already defined in Score_Mapping,
# translating the Excel's qualitative "偏高/中等/偏低" into concrete ranges
# (confirmed with product owner). total_aum range for High_Potential_Invested
# is narrowed so investment_ratio >= 60% remains reachable given
# investment_balance's own 10,000,000 cap.
#
# ARCHETYPE_QUOTAS controls exactly how many of the 1,000 customers are
# generated under each archetype's guidance. "Unconstrained" customers use
# the general Data_Specification distributions with no archetype targeting
# (total_aum_range=None triggers the unconstrained balance-generation path).
ARCHETYPE_QUOTAS = {
    "High_Potential_Gap_Intent": 120,
    "High_Potential_Invested": 120,
    "Investment_Intent_Low_AUM": 120,
    "High_Gap_Low_Intent": 120,
    "Low_Potential": 120,
    "Unconstrained": 400,
}

ARCHETYPE_CONFIG = {
    "High_Potential_Gap_Intent": {
        "total_aum_range": (15_000_000, 42_000_000),
        "twd_demand_ratio_range": (0.40, 0.85),
        "investment_ratio_range": (0.0, 0.10),
        "has_fund": 0,
        "inflow_high": True,
        "fund_view_range": (11, 30),
        "has_abandoned": 1,
    },
    "High_Potential_Invested": {
        "total_aum_range": (15_000_000, 16_600_000),
        "twd_demand_ratio_range": (0.0, 0.10),
        "investment_ratio_range": (0.60, 0.90),
        "has_fund": 1,
        "inflow_high": False,
        "fund_view_range": None,
        "has_abandoned": None,
    },
    "Investment_Intent_Low_AUM": {
        "total_aum_range": (1_000_000, 15_000_000),
        "twd_demand_ratio_range": None,
        "investment_ratio_range": None,
        "has_fund": None,
        "inflow_high": False,
        "fund_view_range": (11, 30),
        "has_abandoned": 1,
    },
    "High_Gap_Low_Intent": {
        "total_aum_range": (15_000_000, 42_000_000),
        "twd_demand_ratio_range": (0.40, 0.85),
        "investment_ratio_range": (0.0, 0.10),
        "has_fund": 0,
        "inflow_high": False,
        "fund_view_range": (0, 1),
        "has_abandoned": 0,
    },
    "Low_Potential": {
        "total_aum_range": (0, 1_000_000),
        "twd_demand_ratio_range": None,
        "investment_ratio_range": (0.0, 0.10),
        "has_fund": None,
        "inflow_high": False,
        "fund_view_range": (0, 1),
        "has_abandoned": 0,
    },
    "Unconstrained": {
        "total_aum_range": None,
        "twd_demand_ratio_range": None,
        "investment_ratio_range": None,
        "has_fund": None,
        "inflow_high": False,
        "fund_view_range": None,
        "has_abandoned": None,
    },
}


def assign_archetypes(n: int, rng: np.random.Generator) -> list[str]:
    """Build the archetype label list per ARCHETYPE_QUOTAS and shuffle order."""
    total = sum(ARCHETYPE_QUOTAS.values())
    assert total == n, f"ARCHETYPE_QUOTAS sums to {total}, expected {n}"
    labels: list[str] = []
    for archetype, quota in ARCHETYPE_QUOTAS.items():
        labels.extend([archetype] * quota)
    rng.shuffle(labels)
    return labels


def generate_demographics(rng: np.random.Generator) -> dict:
    return {
        "occupation": rng.choice(OCCUPATION_CATEGORIES, p=OCCUPATION_WEIGHTS),
        "job_level": rng.choice(JOB_LEVEL_CATEGORIES, p=JOB_LEVEL_WEIGHTS),
        "gender": rng.choice(GENDER_CATEGORIES, p=GENDER_WEIGHTS),
        "age": int(rng.integers(18, 76)),
        "risk_level": rng.choice(RISK_LEVEL_CATEGORIES, p=RISK_LEVEL_WEIGHTS),
    }


def generate_product_holdings(config: dict, rng: np.random.Generator) -> dict:
    has_fund = config["has_fund"]
    if has_fund is None:
        has_fund = int(rng.random() < 0.5)

    return {
        "has_twd_td": int(rng.random() < 0.5),
        "has_fund": int(has_fund),
        "has_fx_deposit": int(rng.random() < 0.5),
        "has_robo": int(rng.random() < 0.15),
        "has_bond": int(rng.random() < 0.15),
        "has_insurance": int(rng.random() < 0.15),
    }


def _sample_ratio(target_range: tuple[float, float] | None, rng: np.random.Generator) -> float | None:
    if target_range is None:
        return None
    low, high = target_range
    return float(rng.uniform(low, high))


def _generate_balances_unconstrained(holdings: dict, rng: np.random.Generator) -> dict:
    """No archetype target: sample each balance independently within its own
    spec range, honoring only the product-holding dependency rules. This is
    the plain Data_Specification GenerationRule behavior (no ratio/tier
    targeting), used for the "Unconstrained" portion of the population.
    """
    has_any_investment_product = any(
        holdings[k] for k in ("has_fund", "has_robo", "has_bond", "has_insurance")
    )

    investment_balance = 0.0
    if has_any_investment_product:
        base = rng.uniform(0, INVESTMENT_MAX)
        if holdings["has_fund"] or holdings["has_bond"]:
            base *= rng.uniform(0.6, 1.0)
        elif holdings["has_robo"] or holdings["has_insurance"]:
            base *= rng.uniform(0.1, 0.4)
        investment_balance = base

    balances = {
        "investment_balance": investment_balance,
        "twd_demand_balance": float(rng.uniform(0, TWD_DEMAND_MAX)),
        "twd_td_balance": float(rng.uniform(0, TWD_TD_MAX)) if holdings["has_twd_td"] else 0.0,
        "fx_demand_balance": float(rng.uniform(0, FX_DEMAND_MAX)) if holdings["has_fx_deposit"] else 0.0,
        "fx_td_balance": float(rng.uniform(0, FX_TD_MAX)) if holdings["has_fx_deposit"] else 0.0,
    }

    total_aum = sum(balances.values())
    return {**balances, "_total_aum": total_aum}


def generate_balances(config: dict, holdings: dict, rng: np.random.Generator) -> dict:
    """Allocate the 5 balance fields so their sum lands in the archetype's
    target total_aum tier, honoring any explicit ratio targets and the
    product-holding flags. total_aum itself is returned only as an internal
    aid and must not be written to the output.

    When config["total_aum_range"] is None (the "Unconstrained" archetype),
    delegates to the plain independent-sampling path instead.
    """
    if config["total_aum_range"] is None:
        return _generate_balances_unconstrained(holdings, rng)

    total_aum_target = float(rng.uniform(*config["total_aum_range"]))

    investment_balance = 0.0
    twd_demand_balance = 0.0

    has_any_investment_product = any(
        holdings[k] for k in ("has_fund", "has_robo", "has_bond", "has_insurance")
    )

    # Fixed components driven by explicit archetype ratio targets.
    if has_any_investment_product:
        inv_ratio = _sample_ratio(config["investment_ratio_range"], rng)
        if inv_ratio is not None:
            investment_balance = min(total_aum_target * inv_ratio, INVESTMENT_MAX)
        else:
            base = rng.uniform(0, INVESTMENT_MAX)
            if holdings["has_fund"] or holdings["has_bond"]:
                base *= rng.uniform(0.6, 1.0)
            elif holdings["has_robo"] or holdings["has_insurance"]:
                base *= rng.uniform(0.1, 0.4)
            investment_balance = min(base, total_aum_target)

    twd_ratio = _sample_ratio(config["twd_demand_ratio_range"], rng)
    if twd_ratio is not None:
        twd_demand_balance = min(total_aum_target * twd_ratio, TWD_DEMAND_MAX)
    else:
        twd_demand_balance = min(rng.uniform(0, TWD_DEMAND_MAX), total_aum_target)

    remaining = max(total_aum_target - investment_balance - twd_demand_balance, 0.0)

    # Randomly split the remainder across td/fx components respecting holding flags.
    free_components = []
    caps = {}
    if holdings["has_twd_td"]:
        free_components.append("twd_td_balance")
        caps["twd_td_balance"] = TWD_TD_MAX
    if holdings["has_fx_deposit"]:
        free_components.append("fx_demand_balance")
        caps["fx_demand_balance"] = FX_DEMAND_MAX
        free_components.append("fx_td_balance")
        caps["fx_td_balance"] = FX_TD_MAX

    balances = {
        "investment_balance": investment_balance,
        "twd_demand_balance": twd_demand_balance,
        "twd_td_balance": 0.0,
        "fx_demand_balance": 0.0,
        "fx_td_balance": 0.0,
    }

    if free_components:
        weights = rng.dirichlet(np.ones(len(free_components)))
        for name, w in zip(free_components, weights):
            balances[name] = min(remaining * w, caps[name])

    if not holdings["has_twd_td"]:
        balances["twd_td_balance"] = 0.0
    if not holdings["has_fx_deposit"]:
        balances["fx_demand_balance"] = 0.0
        balances["fx_td_balance"] = 0.0
    if not has_any_investment_product:
        balances["investment_balance"] = 0.0

    total_aum = sum(balances.values())
    return {**balances, "_total_aum": total_aum}


def generate_cashflow(config: dict, total_aum: float, rng: np.random.Generator) -> dict:
    cap = min(INFLOW_MAX, total_aum * CASHFLOW_TO_AUM_CAP)
    cap = max(cap, 0.0)

    if config["inflow_high"] and cap > 0:
        inflow_1m = rng.uniform(0.6 * cap, cap)
    else:
        inflow_1m = rng.uniform(0, cap) if cap > 0 else 0.0

    outflow_cap = min(OUTFLOW_MAX, total_aum * CASHFLOW_TO_AUM_CAP)
    outflow_cap = max(outflow_cap, 0.0)
    outflow_1m = rng.uniform(0, outflow_cap) if outflow_cap > 0 else 0.0

    return {"inflow_1m": inflow_1m, "outflow_1m": outflow_1m}


def generate_fund_purchase_behavior(total_aum: float, rng: np.random.Generator) -> dict:
    fund_buy_count_1m = int(rng.integers(0, 11))
    if fund_buy_count_1m == 0:
        fund_buy_amount_1m = 0.0
    else:
        scale_cap = min(5_000_000.0, max(100_000.0, total_aum * 0.05))
        fund_buy_amount_1m = float(rng.uniform(0, scale_cap))

    has_rm = int(rng.random() < 0.5)
    return {
        "fund_buy_count_1m": fund_buy_count_1m,
        "fund_buy_amount_1m": fund_buy_amount_1m,
        "has_rm": has_rm,
    }


def generate_digital_behavior(config: dict, rng: np.random.Generator) -> dict:
    view_range = config["fund_view_range"] or (0, 30)
    fund_view_count_1m = int(rng.integers(view_range[0], view_range[1] + 1))

    has_abandoned = config["has_abandoned"]
    if has_abandoned is None:
        has_abandoned = int(rng.random() < 0.10)

    if has_abandoned == 1 and fund_view_count_1m == 0:
        # Dependency rule: has_abandoned_application_1m=1 requires
        # fund_view_count_1m > 0.
        fund_view_count_1m = int(rng.integers(1, view_range[1] + 1 if view_range[1] > 0 else 31))

    return {
        "fund_view_count_1m": fund_view_count_1m,
        "has_abandoned_application_1m": int(has_abandoned),
    }


def generate_customer(customer_id: str, archetype: str, rng: np.random.Generator) -> dict:
    config = ARCHETYPE_CONFIG[archetype]

    row = {"customer_id": customer_id}
    row.update(generate_demographics(rng))

    holdings = generate_product_holdings(config, rng)
    row.update(holdings)

    balances = generate_balances(config, holdings, rng)
    total_aum = balances.pop("_total_aum")
    row.update(balances)

    row.update(generate_cashflow(config, total_aum, rng))
    row.update(generate_fund_purchase_behavior(total_aum, rng))
    row.update(generate_digital_behavior(config, rng))

    return row


def round_amount_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Round all monetary float columns to whole TWD and cast to int64."""
    df = df.copy()
    for col in AMOUNT_COLUMNS:
        df[col] = df[col].round().astype("int64")
    return df


def generate_customers_raw(
    n: int = N_CUSTOMERS, seed: int = RANDOM_SEED, return_archetypes: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    archetypes = assign_archetypes(n, rng)

    rows = []
    for i, archetype in enumerate(archetypes, start=1):
        customer_id = f"C{i:06d}"
        rows.append(generate_customer(customer_id, archetype, rng))

    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    df = round_amount_columns(df)

    if return_archetypes:
        return df, archetypes
    return df


def validate_raw_data(df: pd.DataFrame, n_expected: int = N_CUSTOMERS) -> list[str]:
    """Return a list of validation failure messages (empty if all pass)."""
    errors: list[str] = []

    if len(df) != n_expected:
        errors.append(f"row count = {len(df)}, expected {n_expected}")

    if not df["customer_id"].is_unique:
        errors.append("customer_id is not unique")

    if set(df.columns) != set(RAW_COLUMNS):
        errors.append(f"unexpected columns: {set(df.columns) ^ set(RAW_COLUMNS)}")

    # -- value ranges --
    range_checks = {
        "age": (18, 75),
        "twd_demand_balance": (0, TWD_DEMAND_MAX),
        "twd_td_balance": (0, TWD_TD_MAX),
        "fx_demand_balance": (0, FX_DEMAND_MAX),
        "fx_td_balance": (0, FX_TD_MAX),
        "investment_balance": (0, INVESTMENT_MAX),
        "inflow_1m": (0, INFLOW_MAX),
        "outflow_1m": (0, OUTFLOW_MAX),
        "fund_buy_count_1m": (0, 10),
        "fund_buy_amount_1m": (0, 5_000_000),
        "fund_view_count_1m": (0, 30),
    }
    for col, (low, high) in range_checks.items():
        if not df[col].between(low, high).all():
            errors.append(f"{col} has values outside [{low}, {high}]")

    category_checks = {
        "occupation": set(OCCUPATION_CATEGORIES),
        "job_level": set(JOB_LEVEL_CATEGORIES),
        "gender": set(GENDER_CATEGORIES),
        "risk_level": set(RISK_LEVEL_CATEGORIES),
    }
    for col, allowed in category_checks.items():
        if not set(df[col].unique()).issubset(allowed):
            errors.append(f"{col} has values outside {allowed}")

    bool_cols = [
        "has_twd_td", "has_fund", "has_fx_deposit", "has_robo", "has_bond",
        "has_insurance", "has_rm", "has_abandoned_application_1m",
    ]
    for col in bool_cols:
        if not df[col].isin([0, 1]).all():
            errors.append(f"{col} has values outside {{0, 1}}")

    # -- dependency rules --
    if not (df.loc[df["has_twd_td"] == 0, "twd_td_balance"] == 0).all():
        errors.append("has_twd_td=0 but twd_td_balance != 0 for some rows")

    fx_off = df["has_fx_deposit"] == 0
    if not (df.loc[fx_off, "fx_demand_balance"] == 0).all():
        errors.append("has_fx_deposit=0 but fx_demand_balance != 0 for some rows")
    if not (df.loc[fx_off, "fx_td_balance"] == 0).all():
        errors.append("has_fx_deposit=0 but fx_td_balance != 0 for some rows")

    no_investment_product = (
        (df["has_fund"] == 0) & (df["has_robo"] == 0) &
        (df["has_bond"] == 0) & (df["has_insurance"] == 0)
    )
    if not (df.loc[no_investment_product, "investment_balance"] == 0).all():
        errors.append("all investment product flags=0 but investment_balance != 0 for some rows")

    if not (df.loc[df["fund_buy_count_1m"] == 0, "fund_buy_amount_1m"] == 0).all():
        errors.append("fund_buy_count_1m=0 but fund_buy_amount_1m != 0 for some rows")

    if not (df.loc[df["has_abandoned_application_1m"] == 1, "fund_view_count_1m"] > 0).all():
        errors.append("has_abandoned_application_1m=1 but fund_view_count_1m == 0 for some rows")

    total_aum = (
        df["twd_demand_balance"] + df["twd_td_balance"] +
        df["fx_demand_balance"] + df["fx_td_balance"] + df["investment_balance"]
    )
    epsilon = 5.0  # integer-rounding tolerance across up to 5 summed components (TWD)
    if not (df["inflow_1m"] <= total_aum * CASHFLOW_TO_AUM_CAP + epsilon).all():
        errors.append("inflow_1m exceeds 0.2 x total_aum for some rows")
    if not (df["outflow_1m"] <= total_aum * CASHFLOW_TO_AUM_CAP + epsilon).all():
        errors.append("outflow_1m exceeds 0.2 x total_aum for some rows")

    return errors


def print_qa_checkpoints(df: pd.DataFrame, archetypes: list[str]) -> None:
    """Print quick sanity-check points so a human can eyeball whether the
    generated population looks reasonable. `archetypes` is only used here,
    in memory, for grouping the report -- it is never written to the CSV.
    """
    total_aum = (
        df["twd_demand_balance"] + df["twd_td_balance"] +
        df["fx_demand_balance"] + df["fx_td_balance"] + df["investment_balance"]
    )
    twd_demand_ratio = np.where(total_aum > 0, df["twd_demand_balance"] / total_aum, 0.0)
    investment_ratio = np.where(total_aum > 0, df["investment_balance"] / total_aum, 0.0)

    qa = df.copy()
    qa["archetype"] = archetypes
    qa["_total_aum"] = total_aum
    qa["_twd_demand_ratio"] = twd_demand_ratio
    qa["_investment_ratio"] = investment_ratio

    print("\n=== QA Checkpoint 1: archetype quota (should be 120 x5 + Unconstrained 400) ===")
    print(qa["archetype"].value_counts().reindex(ARCHETYPE_QUOTAS.keys()))

    print("\n=== QA Checkpoint 2: total_aum by archetype (median, TWD) ===")
    print(qa.groupby("archetype")["_total_aum"].median().reindex(ARCHETYPE_QUOTAS.keys()).round(0))

    print("\n=== QA Checkpoint 3: twd_demand_ratio / investment_ratio by archetype (mean) ===")
    print(
        qa.groupby("archetype")[["_twd_demand_ratio", "_investment_ratio"]]
        .mean().reindex(ARCHETYPE_QUOTAS.keys()).round(2)
    )

    print("\n=== QA Checkpoint 4: has_fund rate by archetype ===")
    print(qa.groupby("archetype")["has_fund"].mean().reindex(ARCHETYPE_QUOTAS.keys()).round(2))

    print("\n=== QA Checkpoint 5: fund_view_count_1m / has_abandoned_application_1m by archetype ===")
    print(
        qa.groupby("archetype")[["fund_view_count_1m", "has_abandoned_application_1m"]]
        .mean().reindex(ARCHETYPE_QUOTAS.keys()).round(2)
    )

    print("\n=== QA Checkpoint 6: overall total_aum distribution (all 1,000 customers) ===")
    print(total_aum.describe().round(0))


def main() -> None:
    df, archetypes = generate_customers_raw(return_archetypes=True)

    errors = validate_raw_data(df)
    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Generated {len(df)} customers -> {OUTPUT_PATH}")
    print("Validation: all checks passed")

    print_qa_checkpoints(df, archetypes)


if __name__ == "__main__":
    main()
