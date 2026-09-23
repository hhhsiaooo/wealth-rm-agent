"""Phase 3.12 (WHAT): WHAT Evaluation.

Two parts, both AI evaluation (not software testing -- pytest covers code
correctness with stubs; this measures real retrieval / LLM behaviour):

1. Pipeline evaluation (calls a real LLM + real embeddings): runs
   generate_grounded_next_best_action() for an evaluation set of customers
   and scores each result. The evaluation set is the Top-10 plus the first
   `per_action` customers (by priority rank, from all 1,000) for every action
   the rule layer can select -- Top-10 alone only ever exercises A2/A3.
   WHAT is evaluated in isolation: no WHY insight is passed in.

2. Retrieval evaluation (real embeddings, no LLM): runs the retriever for
   every (category x customer risk level) combination -- including FX, which
   the pipeline never selects (A4 is always outranked) -- so retrieval quality
   is measured independently of action selection.

Metric definitions (docs/ai_product_design.md, "WHAT 評估"). Every rate is
reported with its numerator and denominator (`Rate`), so a reader can see
what it is out of. "raw" = first LLM attempt only (LLM/prompt quality);
"final" = after the grounding policy (what an RM actually sees).

- Groundedness raw/final: validate_next_best_action() passed (per customer).
- Business Rule Compliance (raw output, per customer): action equals the rule
  layer's, action is eligible and is the highest-priority eligible action,
  products only for retrieval actions and of the action's category, and no
  product riskier than the customer.
- Product Retrieval Correctness (per retrieved hit): catalog category matches
  the action's and risk is within the customer's allowed set.
- Product Selection In Retrieved (per picked product).
- Product Feature Accuracy: product_name equals the catalog (per picked
  product) and numeric_claim_support (proxy, see below).
- Product Risk Accuracy: picked product risk <= customer risk (per picked
  product) and risk_statement_accuracy (proxy, see below).
- Recommendation Consistency: same customer generated twice -> same action;
  and Jaccard overlap of the two product picks (where either run has picks).
- Latencies measured directly: data retrieval, product retrieval, LLM
  generation, end-to-end.

Proxy metrics for free text (regex-based, so they can over-flag; they measure
the known limitation of the validator, they are not a gate):
- numeric_claim_support: every number in the explanation / product reasons
  should appear in what the LLM was shown (matched conditions, product data;
  percent and 萬 conversions accepted).
- risk_statement_accuracy: a risk level mentioned in the text should be the
  customer's or a picked product's.

Not claimed: recommendation quality / conversion. There are no outcome
labels; this only measures fidelity to rules and Product Knowledge.

main() calls real LLM/embedding APIs and costs quota -- manual command, never
invoked by pytest.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd
from pydantic import BaseModel

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from agent.latency import TimingLLM as TimedLLMClient  # noqa: E402
from agent.latency import TimingRetriever as TimedRetriever  # noqa: E402
from insight.insight_generator import (  # noqa: E402
    GeminiLLMClient,
    StructuredLLMClient,
    load_env_file,
    select_llm_client,
)
from rag.action_selector import PRIORITY_ORDER, select_action_for_profile  # noqa: E402
from rag.business_rules import (  # noqa: E402
    RETRIEVAL_CATEGORY_BY_ACTION,
    RISK_ORDER,
    allowed_product_risk_levels,
)
from rag.nba_generator import (  # noqa: E402
    NBA_AUDIT_LOG_PATH,
    NBAOutcome,
    _format_product,
    build_query_evidence,
    build_search_query,
    generate_grounded_next_best_action,
)
from rag.product_knowledge import ProductKnowledge  # noqa: E402
from rag.recommendation_validator import default_catalog  # noqa: E402
from rag.retrieval import ProductRetriever  # noqa: E402
from rag.schemas import ActionSelectionResult, NextBestAction  # noqa: E402
from tools.customer_tool import ALL_PATH, TOP10_PATH, get_customer_profile  # noqa: E402

OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"
REPORT_PATH = OUTPUTS_DIR / "what_evaluation_report.csv"
RETRIEVAL_REPORT_PATH = OUTPUTS_DIR / "what_retrieval_report.csv"
SUMMARY_PATH = OUTPUTS_DIR / "what_evaluation_summary.json"
RETRIEVAL_SUMMARY_PATH = OUTPUTS_DIR / "what_retrieval_summary.json"
CHECKPOINT_PATH = OUTPUTS_DIR / "what_evaluation_checkpoint.jsonl"

PER_ACTION_DEFAULT = 3
CONSISTENCY_RUNS = 2

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_IDENTIFIER = re.compile(r"\b(?:C\d{6}|[A-Z]+_\d+|A[1-6])\b")
_RISK_MENTION = re.compile(r"(低|中|高)風險|\b(Low|Medium|High)\b", re.IGNORECASE)
_RISK_ZH = {"低": "Low", "中": "Medium", "高": "High"}


# -- schemas -------------------------------------------------------------------

class Rate(BaseModel):
    """A ratio that always carries its denominator; rate is None when the
    denominator is 0 (metric not applicable), never a fake 0 or 1."""

    numerator: int
    denominator: int
    rate: Optional[float]


def make_rate(numerator: int, denominator: int) -> Rate:
    return Rate(
        numerator=numerator,
        denominator=denominator,
        rate=(numerator / denominator) if denominator else None,
    )


class WhatEvalRecord(BaseModel):
    customer_id: str
    recommended_action: str
    risk_level: str
    status: str
    attempts: int
    is_grounded_initial: bool
    is_grounded_final: bool
    rule_compliant: bool
    retrieval_ran: bool
    retrieved_count: int
    retrieved_correct_count: int
    picked_count: int
    picked_in_retrieved_count: int
    picked_name_correct_count: int
    picked_risk_ok_count: int
    numeric_claims_total: int
    numeric_claims_supported: int
    risk_statements_total: int
    risk_statements_correct: int
    action_consistent: Optional[bool]  # None when the customer was generated only once
    product_jaccard: Optional[float]
    data_retrieval_latency_sec: float
    product_retrieval_latency_sec: float
    llm_generation_latency_sec: float
    end_to_end_latency_sec: float


class WhatEvaluationSummary(BaseModel):
    total_customers: int
    customers_by_action: dict[str, int]
    groundedness_raw: Rate
    groundedness_final: Rate
    business_rule_compliance: Rate
    product_retrieval_correctness: Rate
    product_selection_in_retrieved: Rate
    product_name_accuracy: Rate
    product_risk_accuracy: Rate
    numeric_claim_support: Rate
    risk_statement_accuracy: Rate
    action_consistency: Rate
    product_jaccard_mean: Optional[float]
    product_jaccard_n: int
    retry_rate: Rate
    rejection_rate: Rate
    mean_data_retrieval_latency_sec: float
    mean_product_retrieval_latency_sec: Optional[float]
    mean_llm_generation_latency_sec: float
    mean_end_to_end_latency_sec: float


class RetrievalCaseRecord(BaseModel):
    category: str
    customer_risk_level: str
    currency_hint: Optional[str]
    query: str
    hit_count: int
    category_correct_count: int
    risk_allowed_count: int
    top1_currency_match: Optional[bool]


class RetrievalEvaluationSummary(BaseModel):
    total_cases: int
    empty_result_cases: int
    category_correctness: Rate
    risk_filter_correctness: Rate
    currency_hint_top1_match: Rate


class WhatFullSummary(BaseModel):
    pipeline: WhatEvaluationSummary
    retrieval: RetrievalEvaluationSummary


# TimedRetriever / TimedLLMClient (timing wrappers): defined once in
# agent/latency.py as TimingRetriever / TimingLLM and imported above under
# these names, so the Agent's real-usage latency log and this evaluation
# script measure stages the same way instead of duplicating the logic.

# -- pure scoring helpers (unit-tested with synthetic inputs) -------------------

def _highest_priority_eligible(selection: ActionSelectionResult) -> Optional[str]:
    eligible = {a.action_id for a in selection.evaluated_actions if a.eligible}
    return next((action for action in PRIORITY_ORDER if action in eligible), None)


def check_rule_compliance(
    nba: NextBestAction,
    selection: ActionSelectionResult,
    customer_risk_level: str,
    catalog: dict[str, ProductKnowledge],
) -> bool:
    if nba.recommended_action != selection.recommended_action:
        return False
    selected = next((a for a in selection.evaluated_actions if a.action_id == nba.recommended_action), None)
    if selected is None or not selected.eligible or nba.eligible is not True:
        return False
    if nba.recommended_action != _highest_priority_eligible(selection):
        return False

    expected_category = RETRIEVAL_CATEGORY_BY_ACTION.get(nba.recommended_action)
    if expected_category is None:
        return not nba.products
    allowed = set(allowed_product_risk_levels(customer_risk_level))
    for product in nba.products:
        known = catalog.get(product.product_id)
        if known is None:
            return False
        if known.metadata.product_category != expected_category or known.metadata.risk_level not in allowed:
            return False
    return True


def score_retrieval_hits(
    hits: list[dict[str, Any]],
    expected_category: str,
    allowed_risk_levels: Sequence[str],
    catalog: dict[str, ProductKnowledge],
) -> tuple[int, int]:
    """(correct_count, total): a hit is correct when the catalog says its
    category is the expected one and its risk is within the allowed set."""
    correct = 0
    for hit in hits:
        known = catalog.get(hit["product_id"])
        if (
            known is not None
            and known.metadata.product_category == expected_category
            and known.metadata.risk_level in allowed_risk_levels
        ):
            correct += 1
    return correct, len(hits)


def _format_number(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _number_tokens(text: str) -> set[str]:
    """Distinct numbers in text in canonical form (92.0 == 92, 3.50 == 3.5,
    thousands separators removed)."""
    return {_format_number(float(t)) for t in _NUMBER.findall(text.replace(",", ""))}


def supported_number_universe(texts: Sequence[str]) -> set[str]:
    """Numbers appearing in the material the LLM was shown, plus the
    conversions it may legitimately apply: fraction -> percent, and a whole
    amount -> 萬."""
    universe: set[str] = set()
    for text in texts:
        for token in _number_tokens(_IDENTIFIER.sub(" ", text)):
            universe.add(token)
            value = float(token)
            if 0 < value < 1:
                universe.add(_format_number(value * 100))
            if value >= 10_000:
                universe.add(_format_number(value / 10_000))
    return universe


def score_numeric_claims(claim_texts: Sequence[str], universe: set[str]) -> tuple[int, int]:
    """(supported, total) over the distinct numbers in each claim text."""
    supported = total = 0
    for text in claim_texts:
        for token in _number_tokens(_IDENTIFIER.sub(" ", text)):
            total += 1
            supported += token in universe
    return supported, total


def mentioned_risk_levels(text: str) -> set[str]:
    levels = set()
    for zh, en in _RISK_MENTION.findall(text):
        levels.add(_RISK_ZH[zh] if zh else en.capitalize())
    return levels


def score_risk_statements(
    nba: NextBestAction,
    customer_risk_level: str,
    catalog: dict[str, ProductKnowledge],
) -> tuple[int, int]:
    """(correct, total) over texts that mention a risk level. A product reason
    may mention the customer's risk and that product's; the explanation may
    mention the customer's and any picked product's."""
    picked_levels = {
        catalog[p.product_id].metadata.risk_level for p in nba.products if p.product_id in catalog
    }
    texts: list[tuple[str, set[str]]] = [(nba.explanation, {customer_risk_level} | picked_levels)]
    for product in nba.products:
        known = catalog.get(product.product_id)
        own = {known.metadata.risk_level} if known is not None else set()
        texts.append((product.reason, {customer_risk_level} | own))

    correct = total = 0
    for text, allowed in texts:
        mentioned = mentioned_risk_levels(text)
        if not mentioned:
            continue
        total += 1
        correct += mentioned <= allowed
    return correct, total


def product_jaccard(first: NextBestAction, second: NextBestAction) -> Optional[float]:
    a = {p.product_id for p in first.products}
    b = {p.product_id for p in second.products}
    if not a and not b:
        return None
    return len(a & b) / len(a | b)


def _universe_texts(
    customer_id: str,
    selection: ActionSelectionResult,
    hits: list[dict[str, Any]],
    catalog: dict[str, ProductKnowledge],
    insight_text: Optional[str],
) -> list[str]:
    matched = next(
        a.matched_conditions for a in selection.evaluated_actions if a.action_id == selection.recommended_action
    )
    texts = [customer_id, *matched]
    if insight_text:
        texts.append(insight_text)
    texts.extend(_format_product(catalog[h["product_id"]]) for h in hits if h["product_id"] in catalog)
    return texts


def score_customer(
    customer_id: str,
    risk_level: str,
    outcome: NBAOutcome,
    second_outcome: Optional[NBAOutcome],
    hits: list[dict[str, Any]],
    catalog: dict[str, ProductKnowledge],
    latencies: dict[str, float],
    insight_text: Optional[str] = None,
) -> WhatEvalRecord:
    """Pure: turns one pipeline outcome (raw = first attempt) into a record."""
    selection = outcome.selection
    action_id = selection.recommended_action
    raw = outcome.initial_next_best_action
    expected_category = RETRIEVAL_CATEGORY_BY_ACTION.get(action_id)
    allowed = allowed_product_risk_levels(risk_level)

    retrieved_correct, retrieved_total = (
        score_retrieval_hits(hits, expected_category, allowed, catalog) if expected_category else (0, 0)
    )
    retrieved_ids = {h["product_id"] for h in hits}
    picked = raw.products
    universe = supported_number_universe(_universe_texts(customer_id, selection, hits, catalog, insight_text))
    claims = [raw.explanation, *(p.reason for p in picked)]
    numeric_supported, numeric_total = score_numeric_claims(claims, universe)
    risk_correct, risk_total = score_risk_statements(raw, risk_level, catalog)

    return WhatEvalRecord(
        customer_id=customer_id,
        recommended_action=action_id,
        risk_level=risk_level,
        status=outcome.status,
        attempts=outcome.attempts,
        is_grounded_initial=outcome.initial_check.is_grounded,
        is_grounded_final=outcome.final_check.is_grounded,
        rule_compliant=check_rule_compliance(raw, selection, risk_level, catalog),
        retrieval_ran=expected_category is not None,
        retrieved_count=retrieved_total,
        retrieved_correct_count=retrieved_correct,
        picked_count=len(picked),
        picked_in_retrieved_count=sum(p.product_id in retrieved_ids for p in picked),
        picked_name_correct_count=sum(
            p.product_id in catalog and p.product_name == catalog[p.product_id].content.product_name
            for p in picked
        ),
        picked_risk_ok_count=sum(
            p.product_id in catalog
            and RISK_ORDER[catalog[p.product_id].metadata.risk_level] <= RISK_ORDER[risk_level]
            for p in picked
        ),
        numeric_claims_total=numeric_total,
        numeric_claims_supported=numeric_supported,
        risk_statements_total=risk_total,
        risk_statements_correct=risk_correct,
        action_consistent=(
            None if second_outcome is None else second_outcome.selection.recommended_action == action_id
        ),
        product_jaccard=(
            product_jaccard(raw, second_outcome.initial_next_best_action) if second_outcome is not None else None
        ),
        **latencies,
    )


def compute_summary(records: list[WhatEvalRecord]) -> WhatEvaluationSummary:
    """Pure aggregation -- no I/O, no LLM -- unit-tested with synthetic records."""
    if not records:
        raise ValueError("cannot summarize an empty list of records")

    n = len(records)
    by_action: dict[str, int] = {}
    for r in records:
        by_action[r.recommended_action] = by_action.get(r.recommended_action, 0) + 1

    retrieval_records = [r for r in records if r.retrieval_ran]
    jaccards = [r.product_jaccard for r in records if r.product_jaccard is not None]

    return WhatEvaluationSummary(
        total_customers=n,
        customers_by_action=dict(sorted(by_action.items())),
        groundedness_raw=make_rate(sum(r.is_grounded_initial for r in records), n),
        groundedness_final=make_rate(sum(r.is_grounded_final for r in records), n),
        business_rule_compliance=make_rate(sum(r.rule_compliant for r in records), n),
        product_retrieval_correctness=make_rate(
            sum(r.retrieved_correct_count for r in records), sum(r.retrieved_count for r in records)
        ),
        product_selection_in_retrieved=make_rate(
            sum(r.picked_in_retrieved_count for r in records), sum(r.picked_count for r in records)
        ),
        product_name_accuracy=make_rate(
            sum(r.picked_name_correct_count for r in records), sum(r.picked_count for r in records)
        ),
        product_risk_accuracy=make_rate(
            sum(r.picked_risk_ok_count for r in records), sum(r.picked_count for r in records)
        ),
        numeric_claim_support=make_rate(
            sum(r.numeric_claims_supported for r in records), sum(r.numeric_claims_total for r in records)
        ),
        risk_statement_accuracy=make_rate(
            sum(r.risk_statements_correct for r in records), sum(r.risk_statements_total for r in records)
        ),
        action_consistency=make_rate(
            sum(r.action_consistent is True for r in records),
            sum(r.action_consistent is not None for r in records),
        ),
        product_jaccard_mean=(sum(jaccards) / len(jaccards)) if jaccards else None,
        product_jaccard_n=len(jaccards),
        retry_rate=make_rate(sum(r.attempts == 2 for r in records), n),
        rejection_rate=make_rate(sum(r.status == "rejected" for r in records), n),
        mean_data_retrieval_latency_sec=sum(r.data_retrieval_latency_sec for r in records) / n,
        mean_product_retrieval_latency_sec=(
            sum(r.product_retrieval_latency_sec for r in retrieval_records) / len(retrieval_records)
            if retrieval_records else None
        ),
        mean_llm_generation_latency_sec=sum(r.llm_generation_latency_sec for r in records) / n,
        mean_end_to_end_latency_sec=sum(r.end_to_end_latency_sec for r in records) / n,
    )


# -- evaluation set ------------------------------------------------------------

def select_evaluation_customers(
    all_path: Path = ALL_PATH,
    top10_path: Path = TOP10_PATH,
    per_action: int = PER_ACTION_DEFAULT,
) -> list[str]:
    """Top-10, then the first `per_action` customers by priority rank for each
    selectable action (deterministic; duplicates removed, order kept)."""
    ids = list(pd.read_csv(top10_path)["customer_id"])
    df = pd.read_csv(all_path).sort_values("priority_rank")
    counts: dict[str, int] = {}
    for row in df.to_dict("records"):
        action_id = select_action_for_profile(row).recommended_action
        if counts.get(action_id, 0) < per_action:
            counts[action_id] = counts.get(action_id, 0) + 1
            if row["customer_id"] not in ids:
                ids.append(row["customer_id"])
    return ids


# -- pipeline evaluation (real LLM + retriever) ---------------------------------

def evaluate_customer(
    customer_id: str,
    llm_client: StructuredLLMClient,
    retriever: ProductRetriever,
    source_path: Path = ALL_PATH,
    audit_log_path: Path = NBA_AUDIT_LOG_PATH,
    catalog: dict[str, ProductKnowledge] | None = None,
    consistency_runs: int = CONSISTENCY_RUNS,
    delay_between_runs_sec: float = 0.0,
) -> WhatEvalRecord:
    catalog = catalog if catalog is not None else default_catalog()

    t0 = time.perf_counter()
    profile = get_customer_profile(customer_id, source_path=source_path)
    data_retrieval_latency_sec = time.perf_counter() - t0

    timed_retriever = TimedRetriever(retriever)
    timed_llm = TimedLLMClient(llm_client)
    t1 = time.perf_counter()
    outcome = generate_grounded_next_best_action(
        customer_id, timed_llm, timed_retriever, source_path=source_path,
        audit_log_path=audit_log_path, catalog=catalog,
    )
    end_to_end_latency_sec = time.perf_counter() - t1

    second_outcome: Optional[NBAOutcome] = None
    if consistency_runs > 1:
        if delay_between_runs_sec > 0:
            time.sleep(delay_between_runs_sec)
        second_outcome = generate_grounded_next_best_action(
            customer_id, llm_client, retriever, source_path=source_path,
            audit_log_path=audit_log_path, catalog=catalog,
        )

    return score_customer(
        customer_id=customer_id,
        risk_level=profile["risk_level"],
        outcome=outcome,
        second_outcome=second_outcome,
        hits=timed_retriever.last_hits,
        catalog=catalog,
        latencies={
            "data_retrieval_latency_sec": data_retrieval_latency_sec,
            "product_retrieval_latency_sec": timed_retriever.total_sec,
            "llm_generation_latency_sec": timed_llm.total_sec,
            "end_to_end_latency_sec": end_to_end_latency_sec,
        },
    )


def _load_checkpoint(path: Path) -> dict[str, WhatEvalRecord]:
    if not path.exists():
        return {}
    records = [
        WhatEvalRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {r.customer_id: r for r in records}


def run_pipeline_evaluation(
    customer_ids: Sequence[str],
    llm_client: StructuredLLMClient,
    retriever: ProductRetriever,
    source_path: Path = ALL_PATH,
    audit_log_path: Path = NBA_AUDIT_LOG_PATH,
    consistency_runs: int = CONSISTENCY_RUNS,
    delay_between_calls_sec: float = 0.0,
    checkpoint_path: Optional[Path] = None,
) -> tuple[list[WhatEvalRecord], WhatEvaluationSummary]:
    """Real LLM/embedding calls: 1-2 LLM attempts per run, `consistency_runs`
    runs per customer. delay_between_calls_sec throttles for free-tier rate
    limits (applied between customers and between the repeated runs).

    With checkpoint_path, each finished customer is appended to a JSONL file
    and customers already in it are skipped, so a provider outage midway
    does not discard finished work: rerun and it resumes. An error still
    propagates (a customer is never silently dropped from the denominator)."""
    done = _load_checkpoint(checkpoint_path) if checkpoint_path else {}
    records = []
    ran_any = False
    for i, customer_id in enumerate(customer_ids):
        if customer_id in done:
            print(f"[{i + 1}/{len(customer_ids)}] {customer_id} already in checkpoint, skipping")
            records.append(done[customer_id])
            continue
        if ran_any and delay_between_calls_sec > 0:
            time.sleep(delay_between_calls_sec)
        print(f"[{i + 1}/{len(customer_ids)}] evaluating {customer_id}...", flush=True)
        record = evaluate_customer(
            customer_id, llm_client, retriever, source_path=source_path,
            audit_log_path=audit_log_path, consistency_runs=consistency_runs,
            delay_between_runs_sec=delay_between_calls_sec,
        )
        ran_any = True
        records.append(record)
        if checkpoint_path:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with checkpoint_path.open("a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")
    return records, compute_summary(records)


# -- retrieval evaluation (real retriever, no LLM) -------------------------------

def _retrieval_cases() -> list[tuple[str, str, str, Optional[str]]]:
    """(category, action_id, customer_risk_level, currency_hint). Every
    category x risk level; TD additionally with each single-currency hint."""
    cases: list[tuple[str, str, str, Optional[str]]] = []
    for action_id, category in RETRIEVAL_CATEGORY_BY_ACTION.items():
        for risk_level in RISK_ORDER:
            cases.append((category, action_id, risk_level, None))
            if action_id == "A3":
                cases.append((category, action_id, risk_level, "台幣"))
                cases.append((category, action_id, risk_level, "外幣"))
    return cases


def _synthetic_profile(risk_level: str, currency_hint: Optional[str]) -> dict[str, Any]:
    twd = 1 if currency_hint != "外幣" else 0
    fx = 1 if currency_hint == "外幣" else 0
    return {"risk_level": risk_level, "twd_td_balance": twd, "fx_td_balance": fx}


def evaluate_retrieval(
    retriever: ProductRetriever,
    catalog: dict[str, ProductKnowledge] | None = None,
) -> tuple[list[RetrievalCaseRecord], RetrievalEvaluationSummary]:
    catalog = catalog if catalog is not None else default_catalog()
    records = []
    for category, action_id, risk_level, hint in _retrieval_cases():
        if action_id == "A3" and hint is None:
            profile = {"risk_level": risk_level, "twd_td_balance": 1, "fx_td_balance": 1}
        elif action_id == "A3":
            profile = _synthetic_profile(risk_level, hint)
        else:
            profile = {"risk_level": risk_level}
        query = build_search_query(build_query_evidence(profile, action_id), action_id)
        allowed = allowed_product_risk_levels(risk_level)
        hits = retriever(query, category, allowed)

        category_correct = sum(
            hit["product_id"] in catalog
            and catalog[hit["product_id"]].metadata.product_category == category
            for hit in hits
        )
        risk_allowed = sum(
            hit["product_id"] in catalog and catalog[hit["product_id"]].metadata.risk_level in allowed
            for hit in hits
        )
        top1_match: Optional[bool] = None
        if hint is not None and hits and hits[0]["product_id"] in catalog:
            currency = catalog[hits[0]["product_id"]].metadata.currency
            top1_match = (currency == "TWD") if hint == "台幣" else (currency != "TWD")

        records.append(RetrievalCaseRecord(
            category=category, customer_risk_level=risk_level, currency_hint=hint, query=query,
            hit_count=len(hits), category_correct_count=category_correct,
            risk_allowed_count=risk_allowed, top1_currency_match=top1_match,
        ))
    return records, compute_retrieval_summary(records)


def compute_retrieval_summary(records: list[RetrievalCaseRecord]) -> RetrievalEvaluationSummary:
    total_hits = sum(r.hit_count for r in records)
    currency_cases = [r for r in records if r.top1_currency_match is not None]
    return RetrievalEvaluationSummary(
        total_cases=len(records),
        empty_result_cases=sum(r.hit_count == 0 for r in records),
        category_correctness=make_rate(sum(r.category_correct_count for r in records), total_hits),
        risk_filter_correctness=make_rate(sum(r.risk_allowed_count for r in records), total_hits),
        currency_hint_top1_match=make_rate(sum(bool(r.top1_currency_match) for r in currency_cases), len(currency_cases)),
    )


# -- CLI -----------------------------------------------------------------------

def _write_retrieval_outputs(
    records: list[RetrievalCaseRecord], summary: RetrievalEvaluationSummary
) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r.model_dump() for r in records]).to_csv(
        RETRIEVAL_REPORT_PATH, index=False, encoding="utf-8-sig"
    )
    RETRIEVAL_SUMMARY_PATH.write_text(summary.model_dump_json(indent=2), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Usage:
        python src/evaluation/what_evaluation.py                       # retrieval + full pipeline evaluation
        python src/evaluation/what_evaluation.py --retrieval-only      # embeddings only, no LLM quota used
        python src/evaluation/what_evaluation.py --per-action 2 --consistency-runs 1   # smaller pipeline run

    Needs GEMINI_API_KEY (embeddings + default LLM) in .env. Run it in an
    environment where google-genai and chromadb can share a process. If it
    stops midway (e.g. provider outage or a daily quota), rerun the same
    command: finished customers are kept in
    outputs/what_evaluation_checkpoint.jsonl. Keep the same --per-action /
    --consistency-runs when resuming, or delete the checkpoint to start over."""
    parser = argparse.ArgumentParser(description="Phase 3.12 WHAT evaluation")
    parser.add_argument("--retrieval-only", action="store_true", help="only run the retrieval evaluation (no LLM calls)")
    parser.add_argument("--per-action", type=int, default=PER_ACTION_DEFAULT,
                        help="customers per selectable action, by priority rank (default: %(default)s)")
    parser.add_argument("--consistency-runs", type=int, default=CONSISTENCY_RUNS,
                        help="runs per customer; 1 disables the consistency metric (default: %(default)s)")
    args = parser.parse_args(argv)

    from rag.retrieval import GeminiEmbeddingClient, load_or_build_collection, make_retriever

    load_env_file()
    embedding_client = GeminiEmbeddingClient()
    retriever = make_retriever(load_or_build_collection(embedding_client), embedding_client)

    print("== Retrieval evaluation (no LLM) ==")
    retrieval_records, retrieval_summary = evaluate_retrieval(retriever)
    _write_retrieval_outputs(retrieval_records, retrieval_summary)  # saved now so an LLM failure below cannot lose it
    print(f"Retrieval report -> {RETRIEVAL_REPORT_PATH}\nRetrieval summary -> {RETRIEVAL_SUMMARY_PATH}\n")
    print(retrieval_summary.model_dump_json(indent=2))
    if args.retrieval_only:
        return

    llm_client = select_llm_client()
    # Gemini's free tier is a few requests per minute (observed 5 RPM for
    # a free-tier Gemini model) and a small per-day cap on some models (20 requests/day/model
    # observed); the paid Anthropic tier needs no throttle.
    delay = 13.0 if isinstance(llm_client, GeminiLLMClient) else 0.0

    customer_ids = select_evaluation_customers(per_action=args.per_action)
    print(f"\n== Pipeline evaluation: {len(customer_ids)} customers, {args.consistency_runs} run(s) each ==")
    records, summary = run_pipeline_evaluation(
        customer_ids, llm_client, retriever, delay_between_calls_sec=delay,
        consistency_runs=args.consistency_runs, checkpoint_path=CHECKPOINT_PATH,
    )
    CHECKPOINT_PATH.unlink(missing_ok=True)  # finished: a stale checkpoint would make the next run skip everything

    pd.DataFrame([r.model_dump() for r in records]).to_csv(REPORT_PATH, index=False, encoding="utf-8-sig")
    full = WhatFullSummary(pipeline=summary, retrieval=retrieval_summary)
    SUMMARY_PATH.write_text(full.model_dump_json(indent=2), encoding="utf-8")

    print(f"\nReport -> {REPORT_PATH}\nSummary -> {SUMMARY_PATH}\n")
    print(full.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
