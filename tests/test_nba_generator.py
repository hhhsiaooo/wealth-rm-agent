from __future__ import annotations

import json

import pandas as pd
import pytest

from rag.nba_generator import (
    FALLBACK_MESSAGE,
    NBAGenerationError,
    assemble_next_best_action,
    build_nba_prompt,
    build_query_evidence,
    build_search_query,
    generate_grounded_next_best_action,
    prepare_nba_context,
)
from rag.product_knowledge import PRODUCTS
from rag.schemas import NBALLMOutput, ProductRecommendation
from tools.customer_tool import EXPECTED_FIELDS, CustomerNotFoundError

NAMES = {p.metadata.product_id: p.content.product_name for p in PRODUCTS}


def make_row(customer_id: str, **overrides) -> dict:
    row = {field: 0 for field in EXPECTED_FIELDS}
    row.update({
        "customer_id": customer_id, "risk_level": "High", "has_fund": 0,
        "investment_intent_score": 0.0, "investment_ratio": 0.0,
        "twd_td_balance": 0, "fx_td_balance": 0, "fx_demand_balance": 0, "investment_balance": 0,
    })
    row.update(overrides)
    return row


@pytest.fixture
def csv_path(tmp_path):
    rows = [
        # A2 (also A1, A3 eligible; A2 wins)
        make_row("C000104", investment_intent_score=92.0, twd_td_balance=8_000_000),
        # A1 only
        make_row("C000201", investment_intent_score=60.0, risk_level="Low"),
        # nothing eligible -> A6
        make_row("C000301", risk_level="Low"),
    ]
    path = tmp_path / "customers.csv"
    pd.DataFrame(rows, columns=EXPECTED_FIELDS).to_csv(path, index=False)
    return path


class FakeRetriever:
    def __init__(self, ids=("FUND_007", "FUND_008")):
        self.ids = ids
        self.calls: list[tuple[str, str, list[str] | None]] = []

    def __call__(self, query: str, category: str, risk_levels=None):
        self.calls.append((query, category, risk_levels))
        return [{"product_id": i, "document": "", "metadata": {}, "distance": 0.1} for i in self.ids]


class StubStructuredLLM:
    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.prompts: list[str] = []
        self.models: list[type] = []

    def generate(self, prompt, response_model):
        self.prompts.append(prompt)
        self.models.append(response_model)
        return self._outputs.pop(0)


def good_output(product_id="FUND_007") -> NBALLMOutput:
    return NBALLMOutput(
        explanation="客戶近期投資意圖高，建議介紹基金。",
        products=[ProductRecommendation(product_id=product_id, product_name=NAMES[product_id], reason="可作為候選")],
    )


def bad_output() -> NBALLMOutput:
    return NBALLMOutput(
        explanation="x",
        products=[ProductRecommendation(product_id="FUND_999", product_name="幻覺基金", reason="r")],
    )


# -- search query -------------------------------------------------------------

def test_build_search_query_uses_risk_and_category():
    q = build_search_query(build_query_evidence({"risk_level": "High"}, "A2"), "A2")
    assert "高風險" in q and "基金" in q
    assert "定存" in build_search_query(build_query_evidence({"risk_level": "Low", "twd_td_balance": 1, "fx_td_balance": 1}, "A3"), "A3")
    assert "外幣" in build_search_query(build_query_evidence({"risk_level": "Medium"}, "A4"), "A4")


def test_a3_query_evidence_adds_currency_hint_only_when_unambiguous():
    base = {"risk_level": "Low"}
    assert "台幣" in build_query_evidence({**base, "twd_td_balance": 5, "fx_td_balance": 0}, "A3")
    assert "外幣" in build_query_evidence({**base, "twd_td_balance": 0, "fx_td_balance": 5}, "A3")
    both = build_query_evidence({**base, "twd_td_balance": 5, "fx_td_balance": 5}, "A3")
    assert "台幣" not in both and "外幣" not in both


# -- prepare_nba_context ---------------------------------------------------

def test_context_for_a2_runs_retrieval_with_fund_category(csv_path):
    retriever = FakeRetriever()
    ctx = prepare_nba_context("C000104", retriever, source_path=csv_path)
    assert ctx.selection.recommended_action == "A2"
    assert len(retriever.calls) == 1
    assert retriever.calls[0][1] == "Fund"
    assert [h["product_id"] for h in ctx.hits] == ["FUND_007", "FUND_008"]


def test_context_passes_allowed_risk_levels_to_retriever(csv_path):
    retriever = FakeRetriever()
    ctx = prepare_nba_context("C000104", retriever, source_path=csv_path)  # customer risk = High
    assert retriever.calls[0][2] == ["Low", "Medium", "High"]
    assert ctx.allowed_risk_levels == ["Low", "Medium", "High"]
    assert ctx.query_evidence[0] == "高風險"


def test_context_for_a1_skips_retrieval(csv_path):
    retriever = FakeRetriever()
    ctx = prepare_nba_context("C000201", retriever, source_path=csv_path)
    assert ctx.selection.recommended_action == "A1"
    assert retriever.calls == []
    assert ctx.hits == [] and ctx.search_query is None


def test_context_unknown_customer_propagates(csv_path):
    with pytest.raises(CustomerNotFoundError):
        prepare_nba_context("C999999", FakeRetriever(), source_path=csv_path)


# -- prompt & assembly ---------------------------------------------------------

def test_prompt_contains_rule_result_insight_and_products(csv_path):
    ctx = prepare_nba_context("C000104", FakeRetriever(), insight_text="客戶基金瀏覽28次", source_path=csv_path)
    prompt = build_nba_prompt(ctx)
    assert "A2" in prompt and "基金產品介紹" in prompt
    assert "客戶基金瀏覽28次" in prompt
    assert "FUND_007" in prompt and NAMES["FUND_007"] in prompt
    assert "不可更改" in prompt


def test_prompt_without_insight_says_so(csv_path):
    ctx = prepare_nba_context("C000104", FakeRetriever(), insight_text=None, source_path=csv_path)
    assert "沒有可用的客戶洞察" in build_nba_prompt(ctx)


def test_prompt_for_non_retrieval_action_requires_empty_products(csv_path):
    ctx = prepare_nba_context("C000201", FakeRetriever(), source_path=csv_path)
    prompt = build_nba_prompt(ctx)
    assert "不需要產品檢索" in prompt and "空陣列" in prompt


def test_assemble_copies_rule_layer_fields_and_derives_sources(csv_path):
    ctx = prepare_nba_context("C000104", FakeRetriever(), source_path=csv_path)
    nba = assemble_next_best_action(ctx, good_output())
    assert nba.recommended_action == "A2"
    assert nba.action_name == "基金產品介紹"
    assert nba.eligible is True
    assert nba.sources == ["product_knowledge/FUND_007"]


# -- grounding policy -------------------------------------------------------

def test_grounded_on_first_attempt(csv_path, tmp_path):
    llm = StubStructuredLLM([good_output()])
    outcome = generate_grounded_next_best_action(
        "C000104", llm, FakeRetriever(), source_path=csv_path, audit_log_path=tmp_path / "a.jsonl"
    )
    assert outcome.status == "grounded"
    assert outcome.attempts == 1
    assert outcome.next_best_action is not None
    assert outcome.fallback_message is None
    assert llm.models == [NBALLMOutput]
    assert len(llm.prompts) == 1


def test_corrective_retry_feeds_issues_back(csv_path, tmp_path):
    llm = StubStructuredLLM([bad_output(), good_output()])
    outcome = generate_grounded_next_best_action(
        "C000104", llm, FakeRetriever(), source_path=csv_path, audit_log_path=tmp_path / "a.jsonl"
    )
    assert outcome.status == "grounded_after_retry"
    assert outcome.attempts == 2
    assert outcome.initial_check.is_grounded is False
    assert "FUND_999" in llm.prompts[1]  # the specific issue was fed back


def test_rejected_keeps_rule_layer_result_but_withholds_llm_text(csv_path, tmp_path):
    llm = StubStructuredLLM([bad_output(), bad_output()])
    outcome = generate_grounded_next_best_action(
        "C000104", llm, FakeRetriever(), source_path=csv_path, audit_log_path=tmp_path / "a.jsonl"
    )
    assert outcome.status == "rejected"
    assert outcome.next_best_action is None
    assert outcome.selection.recommended_action == "A2"  # deterministic result still available
    assert outcome.fallback_message == FALLBACK_MESSAGE


def test_non_retrieval_action_with_no_products_is_grounded(csv_path, tmp_path):
    llm = StubStructuredLLM([NBALLMOutput(explanation="建議進行投資需求訪談。", products=[])])
    retriever = FakeRetriever()
    outcome = generate_grounded_next_best_action(
        "C000201", llm, retriever, source_path=csv_path, audit_log_path=tmp_path / "a.jsonl"
    )
    assert outcome.status == "grounded"
    assert outcome.next_best_action.recommended_action == "A1"
    assert retriever.calls == []


def test_non_retrieval_action_recommending_a_product_is_rejected(csv_path, tmp_path):
    llm = StubStructuredLLM([good_output(), good_output()])
    outcome = generate_grounded_next_best_action(
        "C000201", llm, FakeRetriever(), source_path=csv_path, audit_log_path=tmp_path / "a.jsonl"
    )
    assert outcome.status == "rejected"


def test_a6_customer_still_gets_llm_explanation(csv_path, tmp_path):
    llm = StubStructuredLLM([NBALLMOutput(explanation="目前沒有明確聯繫訊號，建議暫不聯繫。", products=[])])
    outcome = generate_grounded_next_best_action(
        "C000301", llm, FakeRetriever(), source_path=csv_path, audit_log_path=tmp_path / "a.jsonl"
    )
    assert outcome.next_best_action.recommended_action == "A6"
    assert len(llm.prompts) == 1


def test_wrong_llm_return_type_raises(csv_path, tmp_path):
    llm = StubStructuredLLM([{"not": "a model"}])
    with pytest.raises(NBAGenerationError):
        generate_grounded_next_best_action(
            "C000104", llm, FakeRetriever(), source_path=csv_path, audit_log_path=tmp_path / "a.jsonl"
        )


def test_audit_log_records_attempts_and_final_status(csv_path, tmp_path):
    log = tmp_path / "a.jsonl"
    llm = StubStructuredLLM([bad_output(), good_output()])
    generate_grounded_next_best_action("C000104", llm, FakeRetriever(), source_path=csv_path, audit_log_path=log)
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [r.get("attempt") for r in records[:2]] == [1, 2]
    assert records[0]["is_grounded"] is False and records[1]["is_grounded"] is True
    assert records[2]["final_status"] == "grounded_after_retry"
