from __future__ import annotations

import pandas as pd
import pytest

from evaluation.what_evaluation import (
    RetrievalCaseRecord,
    TimedLLMClient,
    TimedRetriever,
    WhatEvalRecord,
    check_rule_compliance,
    compute_retrieval_summary,
    compute_summary,
    evaluate_customer,
    evaluate_retrieval,
    make_rate,
    run_pipeline_evaluation,
    mentioned_risk_levels,
    product_jaccard,
    score_numeric_claims,
    score_retrieval_hits,
    score_risk_statements,
    select_evaluation_customers,
    supported_number_universe,
)
from rag.action_selector import select_action_for_profile
from rag.product_knowledge import PRODUCTS
from rag.recommendation_validator import default_catalog
from rag.schemas import NBALLMOutput, NextBestAction, ProductRecommendation
from tools.customer_tool import EXPECTED_FIELDS

CATALOG = default_catalog()
NAMES = {p.metadata.product_id: p.content.product_name for p in PRODUCTS}
RISK = {p.metadata.product_id: p.metadata.risk_level for p in PRODUCTS}


def ids_where(category=None, risk=None, currency=None):
    return [
        p.metadata.product_id for p in PRODUCTS
        if (category is None or p.metadata.product_category == category)
        and (risk is None or p.metadata.risk_level == risk)
        and (currency is None or p.metadata.currency == currency)
    ]


def make_row(customer_id: str, **overrides) -> dict:
    row = {field: 0 for field in EXPECTED_FIELDS}
    row.update({
        "customer_id": customer_id, "risk_level": "High", "has_fund": 0,
        "investment_intent_score": 0.0, "investment_ratio": 0.0,
        "twd_td_balance": 0, "fx_td_balance": 0, "fx_demand_balance": 0, "investment_balance": 0,
    })
    row.update(overrides)
    return row


def pick(product_id: str, reason: str = "可作為候選") -> ProductRecommendation:
    return ProductRecommendation(product_id=product_id, product_name=NAMES[product_id], reason=reason)


def make_nba(action="A2", products=(), explanation="說明", eligible=True) -> NextBestAction:
    return NextBestAction(
        customer_id="C000104", recommended_action=action, action_name="x",
        eligible=eligible, explanation=explanation, products=list(products),
    )


def selection_for(**customer_overrides):
    return select_action_for_profile(make_row("C000104", **customer_overrides))


# -- Rate ------------------------------------------------------------------------

def test_rate_with_zero_denominator_is_none_not_a_fake_number():
    r = make_rate(0, 0)
    assert r.rate is None and r.denominator == 0
    assert make_rate(1, 4).rate == 0.25


# -- business rule compliance ----------------------------------------------------

def test_compliant_a2_output_passes():
    selection = selection_for(investment_intent_score=92.0)  # A2
    fund = ids_where("Fund", "High")[0]
    assert check_rule_compliance(make_nba("A2", [pick(fund)]), selection, "High", CATALOG)


def test_wrong_action_fails():
    selection = selection_for(investment_intent_score=92.0)
    assert not check_rule_compliance(make_nba("A3"), selection, "High", CATALOG)


def test_products_on_non_retrieval_action_fail():
    selection = selection_for(investment_intent_score=60.0, risk_level="Low")  # A1
    assert selection.recommended_action == "A1"
    assert check_rule_compliance(make_nba("A1"), selection, "Low", CATALOG)
    assert not check_rule_compliance(make_nba("A1", [pick(ids_where("Fund")[0])]), selection, "Low", CATALOG)


def test_product_of_wrong_category_fails():
    selection = selection_for(investment_intent_score=92.0)  # A2 -> Fund
    assert not check_rule_compliance(make_nba("A2", [pick(ids_where("TD")[0])]), selection, "High", CATALOG)


def test_product_riskier_than_customer_fails():
    selection = selection_for(investment_intent_score=92.0, risk_level="Medium")
    high_fund = ids_where("Fund", "High")[0]
    assert not check_rule_compliance(make_nba("A2", [pick(high_fund)]), selection, "Medium", CATALOG)


# -- retrieval hit scoring -------------------------------------------------------

def test_score_retrieval_hits_counts_category_and_risk():
    good = ids_where("Fund", "Low")[0]
    wrong_risk = ids_where("Fund", "High")[0]
    wrong_category = ids_where("TD")[0]
    hits = [{"product_id": i} for i in (good, wrong_risk, wrong_category, "NOPE_1")]
    assert score_retrieval_hits(hits, "Fund", ["Low"], CATALOG) == (1, 4)


# -- numeric claims -------------------------------------------------------------

def test_universe_includes_percent_and_wan_conversions():
    universe = supported_number_universe(["investment_ratio=0.12", "twd_td_balance=8000000"])
    assert {"0.12", "12", "8000000", "800"} <= universe


def test_numbers_are_compared_in_canonical_form():
    universe = supported_number_universe(["investment_intent_score=92.0", "年化 3.50%"])
    assert score_numeric_claims(["分數 92，利率 3.5"], universe) == (2, 2)


def test_universe_ignores_identifiers():
    universe = supported_number_universe(["C000104 FUND_007 A2 score=70"])
    assert universe == {"70"}


def test_numeric_claims_flag_unsupported_numbers_but_not_identifiers():
    universe = supported_number_universe(["intent=92"])
    supported, total = score_numeric_claims(["客戶 C000104 意圖 92，建議 A2，預期報酬 8%"], universe)
    assert (supported, total) == (1, 2)  # 92 supported, 8 unsupported


def test_numeric_claims_accept_comma_formatted_amounts():
    universe = supported_number_universe(["twd_td_balance=8000000"])
    assert score_numeric_claims(["定存 8,000,000 元"], universe) == (1, 1)


# -- risk statements ------------------------------------------------------------

def test_mentioned_risk_levels_handles_chinese_and_english():
    assert mentioned_risk_levels("此產品為中風險，客戶為 High") == {"Medium", "High"}
    assert mentioned_risk_levels("沒有提到") == set()


def test_risk_statement_correct_and_wrong():
    medium_fund = ids_where("Fund", "Medium")[0]
    ok = make_nba("A2", [pick(medium_fund, "產品為中風險，客戶為高風險")])
    assert score_risk_statements(ok, "High", CATALOG) == (1, 1)

    wrong = make_nba("A2", [pick(medium_fund, "產品為低風險")])
    assert score_risk_statements(wrong, "High", CATALOG) == (0, 1)


def test_risk_statement_denominator_only_counts_texts_that_mention_risk():
    fund = ids_where("Fund", "High")[0]
    nba = make_nba("A2", [pick(fund, "可作為候選")], explanation="沒有風險字眼")
    assert score_risk_statements(nba, "High", CATALOG) == (0, 0)


# -- consistency ----------------------------------------------------------------

def test_product_jaccard():
    a = make_nba(products=[pick("FUND_007"), pick("FUND_008")])
    b = make_nba(products=[pick("FUND_007")])
    assert product_jaccard(a, b) == 0.5
    assert product_jaccard(a, a) == 1.0
    assert product_jaccard(make_nba(), make_nba()) is None


# -- summary aggregation --------------------------------------------------------

def make_record(**overrides) -> WhatEvalRecord:
    data = dict(
        customer_id="C1", recommended_action="A2", risk_level="High", status="grounded", attempts=1,
        is_grounded_initial=True, is_grounded_final=True, rule_compliant=True, retrieval_ran=True,
        retrieved_count=3, retrieved_correct_count=3, picked_count=2, picked_in_retrieved_count=2,
        picked_name_correct_count=2, picked_risk_ok_count=2, numeric_claims_total=4,
        numeric_claims_supported=3, risk_statements_total=2, risk_statements_correct=2,
        action_consistent=True, product_jaccard=1.0, data_retrieval_latency_sec=0.1,
        product_retrieval_latency_sec=0.5, llm_generation_latency_sec=2.0, end_to_end_latency_sec=2.6,
    )
    data.update(overrides)
    return WhatEvalRecord(**data)


def test_compute_summary_rates_carry_denominators():
    records = [
        make_record(),
        make_record(
            customer_id="C2", recommended_action="A1", retrieval_ran=False, retrieved_count=0,
            retrieved_correct_count=0, picked_count=0, picked_in_retrieved_count=0,
            picked_name_correct_count=0, picked_risk_ok_count=0, numeric_claims_total=0,
            numeric_claims_supported=0, risk_statements_total=0, risk_statements_correct=0,
            is_grounded_initial=False, attempts=2, status="grounded_after_retry",
            product_jaccard=None, product_retrieval_latency_sec=0.0,
        ),
    ]
    s = compute_summary(records)
    assert s.total_customers == 2
    assert s.customers_by_action == {"A1": 1, "A2": 1}
    assert (s.groundedness_raw.numerator, s.groundedness_raw.denominator) == (1, 2)
    assert s.groundedness_final.rate == 1.0
    assert s.product_name_accuracy.denominator == 2  # only customer 1 picked products
    assert s.product_retrieval_correctness.denominator == 3
    assert s.numeric_claim_support.rate == 0.75
    assert s.retry_rate.numerator == 1
    assert s.product_jaccard_mean == 1.0 and s.product_jaccard_n == 1
    # product retrieval latency averages over customers where retrieval ran only
    assert s.mean_product_retrieval_latency_sec == 0.5


def test_compute_summary_not_applicable_metrics_are_none():
    only_a1 = make_record(
        recommended_action="A1", retrieval_ran=False, retrieved_count=0, retrieved_correct_count=0,
        picked_count=0, picked_in_retrieved_count=0, picked_name_correct_count=0,
        picked_risk_ok_count=0, product_jaccard=None,
    )
    s = compute_summary([only_a1])
    assert s.product_name_accuracy.rate is None
    assert s.product_retrieval_correctness.rate is None
    assert s.mean_product_retrieval_latency_sec is None
    assert s.product_jaccard_mean is None


def test_compute_summary_action_consistency_ignores_single_run_customers():
    s = compute_summary([make_record(action_consistent=None), make_record(action_consistent=False)])
    assert (s.action_consistency.numerator, s.action_consistency.denominator) == (0, 1)


def test_compute_summary_rejects_empty():
    with pytest.raises(ValueError):
        compute_summary([])


# -- evaluation set ------------------------------------------------------------

def test_select_evaluation_customers_top10_plus_first_n_per_action(tmp_path):
    rows = [
        make_row("C000001", investment_intent_score=92.0, priority_rank=1),   # A2
        make_row("C000002", investment_intent_score=92.0, priority_rank=2),   # A2
        make_row("C000003", investment_intent_score=92.0, priority_rank=3),   # A2 (over the limit)
        make_row("C000004", investment_intent_score=60.0, risk_level="Low", priority_rank=4),  # A1
        make_row("C000005", risk_level="Low", priority_rank=5),               # A6
    ]
    all_path, top10_path = tmp_path / "all.csv", tmp_path / "top10.csv"
    pd.DataFrame(rows, columns=EXPECTED_FIELDS).to_csv(all_path, index=False)
    pd.DataFrame(rows[:1], columns=EXPECTED_FIELDS).to_csv(top10_path, index=False)

    ids = select_evaluation_customers(all_path, top10_path, per_action=2)
    assert ids == ["C000001", "C000002", "C000004", "C000005"]  # top10 first, C000003 cut, no duplicates


# -- timing wrappers ------------------------------------------------------------

def test_timed_wrappers_record_calls():
    retriever = TimedRetriever(lambda q, c, r=None: [{"product_id": "FUND_007"}])
    assert retriever("q", "Fund", ["Low"]) == [{"product_id": "FUND_007"}]
    assert retriever.last_hits == [{"product_id": "FUND_007"}] and retriever.total_sec >= 0

    class Client:
        def generate(self, prompt, response_model):
            return "ok"

    llm = TimedLLMClient(Client())
    assert llm.generate("p", NBALLMOutput) == "ok" and llm.total_sec >= 0


# -- evaluate_customer with stubs -----------------------------------------------

class StubLLM:
    def __init__(self, outputs):
        self._outputs = list(outputs)

    def generate(self, prompt, response_model):
        return self._outputs.pop(0)


class CatalogRetriever:
    """In-memory stand-in for the vector search: honours category + risk filter."""

    def __call__(self, query, category, risk_levels=None):
        return [
            {"product_id": p.metadata.product_id, "document": "", "metadata": {}, "distance": 0.1}
            for p in PRODUCTS
            if p.metadata.product_category == category
            and (risk_levels is None or p.metadata.risk_level in risk_levels)
        ][:3]


def test_evaluate_customer_scores_a_grounded_a2_run(tmp_path):
    csv_path = tmp_path / "customers.csv"
    pd.DataFrame(
        [make_row("C000104", investment_intent_score=92.0, twd_td_balance=8_000_000)],
        columns=EXPECTED_FIELDS,
    ).to_csv(csv_path, index=False)
    retrieved = CatalogRetriever()("q", "Fund", ["Low", "Medium", "High"])
    chosen = retrieved[0]["product_id"]

    def output():
        return NBALLMOutput(
            explanation="客戶投資意圖分數 92，建議介紹基金。",
            products=[pick(chosen, f"風險為{'低中高'[['Low', 'Medium', 'High'].index(RISK[chosen])]}風險，可作為候選")],
        )

    record = evaluate_customer(
        "C000104", StubLLM([output(), output()]), CatalogRetriever(), source_path=csv_path,
        audit_log_path=tmp_path / "audit.jsonl", consistency_runs=2,
    )
    assert record.recommended_action == "A2"
    assert record.status == "grounded" and record.is_grounded_initial and record.rule_compliant
    assert record.retrieval_ran and record.retrieved_count == 3 and record.retrieved_correct_count == 3
    assert record.picked_count == record.picked_in_retrieved_count == record.picked_name_correct_count == 1
    assert record.picked_risk_ok_count == 1
    assert record.numeric_claims_total == 0 or record.numeric_claims_supported == record.numeric_claims_total
    assert record.action_consistent is True and record.product_jaccard == 1.0
    assert record.product_retrieval_latency_sec >= 0 and record.llm_generation_latency_sec >= 0
    assert record.end_to_end_latency_sec >= record.llm_generation_latency_sec


def test_evaluate_customer_single_run_leaves_consistency_none(tmp_path):
    csv_path = tmp_path / "customers.csv"
    pd.DataFrame([make_row("C000301", risk_level="Low")], columns=EXPECTED_FIELDS).to_csv(csv_path, index=False)
    record = evaluate_customer(
        "C000301", StubLLM([NBALLMOutput(explanation="暫不聯繫。", products=[])]), CatalogRetriever(),
        source_path=csv_path, audit_log_path=tmp_path / "audit.jsonl", consistency_runs=1,
    )
    assert record.recommended_action == "A6" and not record.retrieval_ran
    assert record.action_consistent is None and record.product_jaccard is None


# -- retrieval evaluation -------------------------------------------------------

def test_evaluate_retrieval_covers_all_categories_and_risk_levels():
    records, summary = evaluate_retrieval(CatalogRetriever())
    assert summary.total_cases == 15  # 3 categories x 3 risk levels + TD x 3 risks x 2 currency hints
    assert {r.category for r in records} == {"Fund", "TD", "FX"}
    assert summary.category_correctness.rate == 1.0
    assert summary.risk_filter_correctness.rate == 1.0
    assert summary.empty_result_cases == 0


def test_evaluate_retrieval_detects_a_retriever_that_ignores_the_risk_filter():
    def leaky(query, category, risk_levels=None):
        return CatalogRetriever()(query, category, None)

    _, summary = evaluate_retrieval(leaky)
    assert summary.risk_filter_correctness.rate < 1.0


def test_currency_hint_top1_match_is_scored_from_catalog_currency():
    def twd_first(query, category, risk_levels=None):
        hits = CatalogRetriever()(query, category, risk_levels)
        return sorted(hits, key=lambda h: CATALOG[h["product_id"]].metadata.currency != "TWD")

    records, summary = evaluate_retrieval(twd_first)
    twd_cases = [r for r in records if r.currency_hint == "台幣"]
    fx_cases = [r for r in records if r.currency_hint == "外幣"]
    assert twd_cases and all(r.top1_currency_match for r in twd_cases)
    assert fx_cases and not any(r.top1_currency_match for r in fx_cases)
    assert summary.currency_hint_top1_match.denominator == len(twd_cases) + len(fx_cases)


def test_retrieval_summary_with_no_hits_reports_empty_cases():
    record = RetrievalCaseRecord(
        category="Fund", customer_risk_level="Low", currency_hint=None, query="q", hit_count=0,
        category_correct_count=0, risk_allowed_count=0, top1_currency_match=None,
    )
    summary = compute_retrieval_summary([record])
    assert summary.empty_result_cases == 1
    assert summary.category_correctness.rate is None


# -- checkpoint / resume --------------------------------------------------------

def test_pipeline_evaluation_resumes_from_checkpoint_after_a_failure(tmp_path):
    csv_path = tmp_path / "customers.csv"
    pd.DataFrame(
        [make_row("C000301", risk_level="Low"), make_row("C000302", risk_level="Low")],
        columns=EXPECTED_FIELDS,
    ).to_csv(csv_path, index=False)
    checkpoint = tmp_path / "checkpoint.jsonl"
    output = lambda: NBALLMOutput(explanation="暫不聯繫。", products=[])  # noqa: E731

    class OutageAfterOne:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt, response_model):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("503 UNAVAILABLE")
            return output()

    with pytest.raises(RuntimeError):
        run_pipeline_evaluation(
            ["C000301", "C000302"], OutageAfterOne(), CatalogRetriever(), source_path=csv_path,
            audit_log_path=tmp_path / "audit.jsonl", consistency_runs=1, checkpoint_path=checkpoint,
        )
    assert len(checkpoint.read_text(encoding="utf-8").splitlines()) == 1  # first customer survived

    resumed = StubLLM([output()])  # only ONE call is available: the finished customer must not be re-run
    records, summary = run_pipeline_evaluation(
        ["C000301", "C000302"], resumed, CatalogRetriever(), source_path=csv_path,
        audit_log_path=tmp_path / "audit.jsonl", consistency_runs=1, checkpoint_path=checkpoint,
    )
    assert [r.customer_id for r in records] == ["C000301", "C000302"]
    assert summary.total_customers == 2
