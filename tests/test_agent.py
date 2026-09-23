from __future__ import annotations

import re

import pandas as pd
import pytest

from agent.agent_llm import (
    ToolExecution,
    AgentDecision,
    StructuredAgentLLM,
    ToolCallRequest,
    build_context,
    build_structured_prompt,
)
from agent.rm_agent import DEFAULT_CLARIFICATION, Agent, summarize_result
from agent.render import DISCLAIMER, render_response
from agent.agent_tools import (
    TOOL_INSIGHT,
    TOOL_NBA,
    TOOL_TOP10,
    AgentTools,
    LazyRetriever,
    SessionState,
    normalize_customer_id,
)
from insight.insight_generator import CustomerInsight, KeyField
from rag.product_knowledge import PRODUCTS
from rag.schemas import NBALLMOutput, ProductRecommendation
from tools.customer_tool import EXPECTED_FIELDS

NAMES = {p.metadata.product_id: p.content.product_name for p in PRODUCTS}


def make_row(customer_id: str, rank: int, **overrides) -> dict:
    row = {field: 0 for field in EXPECTED_FIELDS}
    row.update({
        "customer_id": customer_id, "risk_level": "High", "has_fund": 0,
        "investment_intent_score": 0.0, "investment_ratio": 0.0,
        "twd_td_balance": 0, "fx_td_balance": 0, "fx_demand_balance": 0, "investment_balance": 0,
        "priority_rank": rank, "priority_score": 90.5,
        "asset_potential_score": 80.0, "investment_gap_score": 94.0,
    })
    row.update(overrides)
    return row


@pytest.fixture
def paths(tmp_path):
    rows = [
        make_row("C000104", 1, investment_intent_score=92.0),            # A2
        make_row("C000518", 2, investment_intent_score=92.0),            # A2
        make_row("C000301", 3, risk_level="Low"),                        # A6
    ]
    all_path, top10_path = tmp_path / "all.csv", tmp_path / "top10.csv"
    df = pd.DataFrame(rows, columns=EXPECTED_FIELDS)
    df.to_csv(all_path, index=False)
    df.iloc[:2].to_csv(top10_path, index=False)
    return {"all": all_path, "top10": top10_path, "tmp": tmp_path}


class FakeBackend:
    """Stands in for the Gemini/Anthropic client: __call__ = WHY, generate = WHAT."""

    def __init__(self, rank_by_id=None, insight_bad=False):
        self.calls: list[str] = []
        self.insight_bad = insight_bad
        self.rank_by_id = rank_by_id or {"C000104": 1, "C000518": 2, "C000301": 3}
        self.nba_prompts: list[str] = []

    def __call__(self, prompt):
        cid = re.search(r"C\d{6}", prompt).group(0)
        self.calls.append(f"insight:{cid}")
        field = "no_such_field" if self.insight_bad else "risk_level"
        return CustomerInsight(
            customer_id=cid, priority_rank=self.rank_by_id[cid], priority_score=90.5,
            key_fields=[KeyField(field=field, value="High")], insight="客戶投資意圖高、尚未持有基金。",
        )

    def generate(self, prompt, response_model):
        cid = re.search(r"C\d{6}", prompt).group(0)
        self.calls.append(f"nba:{cid}")
        self.nba_prompts.append(prompt)
        products = []
        if "FUND_007" in prompt:
            products = [ProductRecommendation(product_id="FUND_007", product_name=NAMES["FUND_007"], reason="可作為候選")]
        return NBALLMOutput(explanation="建議介紹基金。", products=products)


def fake_retriever(query, category, risk_levels=None):
    return [{"product_id": "FUND_007", "document": "", "metadata": {}, "distance": 0.1}]


class ScriptedRouter:
    """Stands in for the LLM side (AgentLLMClient): start() / next(executions)."""

    def __init__(self, decisions):
        self._decisions = list(decisions)
        self.contexts: list[str] = []          # one per question (start)
        self.received: list[list] = []         # executions handed to each next() call

    @property
    def calls(self) -> int:
        return len(self.received)

    def start(self, context):
        self.contexts.append(context)

    def next(self, executions):
        self.received.append(list(executions))
        if isinstance(self._decisions[0], Exception):
            raise self._decisions.pop(0)
        return self._decisions.pop(0) if len(self._decisions) > 1 else self._decisions[0]


def decision(*calls, reply="", more=False) -> AgentDecision:
    return AgentDecision(
        tool_calls=[ToolCallRequest(tool_name=n, customer_id=c) for n, c in calls],
        reply=reply, needs_more_steps=more,
    )


@pytest.fixture
def backend():
    return FakeBackend()


@pytest.fixture
def tools(paths, backend):
    return AgentTools(
        backend, fake_retriever, source_path=paths["all"], top10_path=paths["top10"],
        insight_audit_log_path=paths["tmp"] / "insight.jsonl", nba_audit_log_path=paths["tmp"] / "nba.jsonl",
    )


def make_agent(tools, decisions) -> tuple[Agent, ScriptedRouter]:
    router = ScriptedRouter(decisions)
    return Agent(router, tools), router


# -- tools -----------------------------------------------------------------------

def test_top10_tool_lists_ranked_customers(tools):
    result = tools.run(TOOL_TOP10, None, SessionState())
    assert result.ok and [r.customer_id for r in result.top10] == ["C000104", "C000518"]
    assert result.top10[0].priority_rank == 1 and result.top10[0].investment_gap_score == 94.0


def test_customer_id_is_only_case_normalized():
    assert normalize_customer_id("  c000104 ") == "C000104"
    assert normalize_customer_id("C0001") == "C0001"  # never repaired or padded
    assert normalize_customer_id(None) == ""


def test_insight_tool_generates_then_reuses_the_cached_insight(tools, backend):
    session = SessionState()
    first = tools.run(TOOL_INSIGHT, " c000104 ", session)
    second = tools.run(TOOL_INSIGHT, "C000104", session)
    assert first.ok and first.insight.status == "grounded" and second.insight.status == "grounded"
    assert backend.calls == ["insight:C000104"]  # the second question did not cost another LLM call


def test_insight_tool_retries_a_previously_rejected_insight(paths):
    bad = FakeBackend(insight_bad=True)
    t = AgentTools(bad, fake_retriever, source_path=paths["all"], top10_path=paths["top10"],
                   insight_audit_log_path=paths["tmp"] / "i.jsonl", nba_audit_log_path=paths["tmp"] / "n.jsonl")
    session = SessionState()
    assert t.run(TOOL_INSIGHT, "C000104", session).insight.status == "rejected"
    calls_after_first = len(bad.calls)
    t.run(TOOL_INSIGHT, "C000104", session)
    assert len(bad.calls) > calls_after_first


def test_nba_runs_why_first_when_session_has_no_insight(tools, backend):
    session = SessionState()
    result = tools.run(TOOL_NBA, "C000104", session)
    assert result.ok and result.insight_source == "generated"
    assert backend.calls == ["insight:C000104", "nba:C000104"]  # WHY strictly before WHAT
    assert "客戶投資意圖高、尚未持有基金。" in backend.nba_prompts[0]  # the insight reached the WHAT prompt
    assert session.insights["C000104"].status == "grounded"


def test_nba_reuses_cached_insight_without_running_why_again(tools, backend):
    session = SessionState()
    tools.run(TOOL_INSIGHT, "C000104", session)
    result = tools.run(TOOL_NBA, "C000104", session)
    assert result.insight_source == "cached"
    assert backend.calls == ["insight:C000104", "nba:C000104"]


def test_nba_still_runs_without_insight_when_why_is_rejected(paths):
    bad = FakeBackend(insight_bad=True)
    t = AgentTools(bad, fake_retriever, source_path=paths["all"], top10_path=paths["top10"],
                   insight_audit_log_path=paths["tmp"] / "i.jsonl", nba_audit_log_path=paths["tmp"] / "n.jsonl")
    result = t.run(TOOL_NBA, "C000104", SessionState())
    assert result.ok and result.insight_source == "unavailable"
    assert result.nba.selection.recommended_action == "A2"
    assert "沒有可用的客戶洞察" in bad.nba_prompts[0]


def test_unknown_customer_is_an_error_and_runs_no_llm(tools, backend):
    result = tools.run(TOOL_NBA, "C999999", SessionState())
    assert not result.ok and "C999999" in result.error
    assert backend.calls == []


def test_missing_customer_id_is_an_error(tools):
    result = tools.run(TOOL_INSIGHT, "  ", SessionState())
    assert not result.ok and "客戶編號" in result.error


def test_tool_outside_the_allowlist_is_refused(tools):
    result = tools.run("delete_all_customers", "C000104", SessionState())
    assert not result.ok and "不支援" in result.error


def test_failure_inside_a_tool_becomes_an_error_result_not_an_exception(paths):
    class Broken(FakeBackend):
        def __call__(self, prompt):
            raise RuntimeError("503 UNAVAILABLE")

    t = AgentTools(Broken(), fake_retriever, source_path=paths["all"], top10_path=paths["top10"],
                   insight_audit_log_path=paths["tmp"] / "i.jsonl", nba_audit_log_path=paths["tmp"] / "n.jsonl")
    result = t.run(TOOL_INSIGHT, "C000104", SessionState())
    assert not result.ok and "503" in result.error


def test_lazy_retriever_builds_only_on_first_use():
    built = []

    def factory():
        built.append(1)
        return fake_retriever

    lazy = LazyRetriever(factory)
    assert built == []
    lazy("q", "Fund", ["High"])
    lazy("q", "Fund", ["High"])
    assert built == [1]


# -- agent loop ------------------------------------------------------------------

def test_single_round_question_calls_the_router_once(tools):
    agent, router = make_agent(tools, [decision((TOOL_TOP10, ""))])
    response = agent.ask("今天要先聯繫誰？")
    assert router.calls == 1  # needs_more_steps=False: no extra LLM call to "finish"
    assert "【今日優先名單】" in response.reply and "C000104" in response.reply
    assert DISCLAIMER in response.reply
    assert [r.tool for r in response.tool_results] == [TOOL_TOP10]


def test_clarification_is_the_llm_message_when_no_tool_runs(tools, backend):
    agent, _ = make_agent(tools, [decision(reply="請提供客戶編號。")])
    response = agent.ask("他下一步怎麼做？")
    assert response.reply == "請提供客戶編號。" and response.tool_results == [] and backend.calls == []


def test_empty_llm_reply_falls_back_to_the_default_help_text(tools):
    agent, _ = make_agent(tools, [decision(reply="   ")])
    assert agent.ask("你好").reply == DEFAULT_CLARIFICATION


def test_llm_free_text_is_ignored_once_a_tool_has_run(tools):
    agent, _ = make_agent(tools, [decision((TOOL_TOP10, ""), reply="這位客戶一定會買基金，保證獲利！")])
    response = agent.ask("今天要先聯繫誰？")
    assert "保證獲利" not in response.reply


def test_multi_step_question_uses_earlier_results_to_pick_the_customer(tools, backend):
    agent, router = make_agent(tools, [
        decision((TOOL_TOP10, ""), more=True),
        decision((TOOL_NBA, "C000104")),
    ])
    response = agent.ask("名單第一名的客戶，下一步怎麼做？")
    assert router.calls == 2
    round2 = router.received[1]
    assert len(round2) == 1 and round2[0].call.tool_name == TOOL_TOP10
    assert "第1名 C000104" in round2[0].summary  # round 2 saw round 1's result summary
    assert [r.tool for r in response.tool_results] == [TOOL_TOP10, TOOL_NBA]
    assert "【今日優先名單】" in response.reply and "【行動建議】C000104" in response.reply
    assert backend.calls == ["insight:C000104", "nba:C000104"]


def test_insight_result_ends_the_question_without_another_llm_call(tools):
    # Only the Top-10 result can inform a further call, so an insight round never costs a follow-up LLM call.
    agent, router = make_agent(tools, [decision((TOOL_INSIGHT, "C000104"), more=True), decision()])
    agent.ask("看 C000104 的洞察")
    assert router.calls == 1


def test_llm_gets_summaries_not_insight_text(tools):
    agent, router = make_agent(tools, [
        decision((TOOL_TOP10, ""), (TOOL_INSIGHT, "C000104"), more=True), decision(),
    ])
    agent.ask("列名單並看第一名洞察")
    summaries = " ".join(e.summary for e in router.received[1])
    assert "洞察狀態：grounded" in summaries and "第1名 C000104" in summaries
    assert "客戶投資意圖高" not in summaries  # the LLM never sees the content it must not rewrite


def test_duplicate_calls_are_skipped_but_still_answered(tools, backend):
    agent, router = make_agent(tools, [
        decision((TOOL_TOP10, ""), (TOOL_TOP10, ""), more=True), decision(),
    ])
    response = agent.ask("名單")
    assert len(response.tool_results) == 1 and any("重複" in line for line in response.trace)
    # Native function calling needs one response per call: the skipped call is answered too.
    assert len(router.received[1]) == 2 and "重複" in router.received[1][1].summary


def test_tool_call_cap_stops_a_runaway_plan(tools):
    many = [(TOOL_INSIGHT, cid) for cid in ("C000104", "C000518", "C000301", "C000104", "C000518", "C000301")]
    router = ScriptedRouter([decision(*many, more=True)])
    agent = Agent(router, tools, max_tool_calls=2)
    response = agent.ask("全部")
    assert len(response.tool_results) == 2
    assert any("上限" in line for line in response.trace)


def test_step_cap_stops_before_another_round(tools):
    router = ScriptedRouter([decision((TOOL_TOP10, ""), more=True), decision((TOOL_NBA, "C000104"))])
    agent = Agent(router, tools, max_steps=1)
    response = agent.ask("名單第一名怎麼做")
    assert router.calls == 1 and len(response.tool_results) == 1
    assert any("最多 1 輪" in line for line in response.trace)


def test_repeating_router_ends_when_nothing_new_runs(tools):
    router = ScriptedRouter([decision((TOOL_TOP10, ""), more=True)])  # returns the same decision forever
    agent = Agent(router, tools)
    response = agent.ask("查")
    assert len(response.tool_results) == 1 and router.calls == 2


def test_unknown_tool_from_the_llm_is_refused_and_reported(tools):
    agent, _ = make_agent(tools, [decision(("transfer_money", "C000104"))])
    response = agent.ask("幫我轉帳")
    assert "【無法完成】" in response.reply and "不支援" in response.reply
    assert any("白名單" in line for line in response.trace)


def test_router_llm_failure_is_reported_not_raised(tools):
    agent, _ = make_agent(tools, [RuntimeError("429 quota")])
    response = agent.ask("今天要先聯繫誰？")
    assert "無法連線" in response.reply and "429" in response.reply


def test_router_failure_in_a_later_round_still_shows_earlier_results(tools):
    router = ScriptedRouter([decision((TOOL_TOP10, ""), more=True), RuntimeError("503")])
    response = Agent(router, tools).ask("名單第一名怎麼做？")
    assert "【今日優先名單】" in response.reply
    assert any("LLM 判斷失敗" in line for line in response.trace)


def test_unknown_customer_shows_an_error_and_the_conversation_continues(tools):
    agent, _ = make_agent(tools, [decision((TOOL_INSIGHT, "C999999"))])
    response = agent.ask("C999999 為什麼值得聯繫？")
    assert "【無法完成】" in response.reply and DISCLAIMER not in response.reply
    assert agent.ask("再問一次").reply  # still usable


def test_second_question_is_told_which_customers_were_discussed(tools):
    agent, router = make_agent(tools, [decision((TOOL_INSIGHT, "C000104")), decision((TOOL_NBA, "C000104"))])
    agent.ask("C000104 為什麼值得聯繫？")
    agent.ask("那他下一步呢？")
    second_context = router.contexts[1]
    assert "C000104（已有洞察）" in second_context
    assert "C000104 為什麼值得聯繫？" in second_context  # recent question kept as context


def test_reset_clears_session_state(tools):
    agent, _ = make_agent(tools, [decision((TOOL_INSIGHT, "C000104"))])
    agent.ask("C000104 為什麼值得聯繫？")
    assert "C000104" in agent.session.insights
    agent.reset()
    assert agent.session.insights == {} and agent.session.discussed == []


def test_on_event_streams_the_trace(tools):
    agent, _ = make_agent(tools, [decision((TOOL_TOP10, ""))])
    seen: list[str] = []
    response = agent.ask("名單", on_event=seen.append)
    assert seen == response.trace and any(TOOL_TOP10 in line for line in seen)


def test_structured_agent_llm_adapter_asks_for_an_agent_decision_and_accumulates_summaries():
    class Client:
        def __init__(self):
            self.prompts = []

        def generate(self, prompt, response_model):
            self.prompts.append(prompt)
            self.model = response_model
            return decision()

    client = Client()
    adapter = StructuredAgentLLM(client)
    adapter.start("CONTEXT-ONE")
    adapter.next([])
    adapter.next([ToolExecution(call=ToolCallRequest(tool_name=TOOL_TOP10, customer_id=""), summary="SUMMARY-A")])
    assert client.model is AgentDecision
    assert "CONTEXT-ONE" in client.prompts[0] and "SUMMARY-A" not in client.prompts[0]
    assert "SUMMARY-A" in client.prompts[1]
    adapter.start("CONTEXT-TWO")  # a new question starts clean
    adapter.next([])
    assert "SUMMARY-A" not in client.prompts[2] and "CONTEXT-TWO" in client.prompts[2]


# -- rendering (code-written reply) ------------------------------------------------

def test_nba_reply_shows_rule_result_products_and_insight_source(tools):
    response = Agent(ScriptedRouter([decision((TOOL_NBA, "C000104"))]), tools).ask("C000104 下一步？")
    reply = response.reply
    assert "建議行動：A2 基金產品介紹" in reply
    assert "investment_intent_score=92.0 >= 70" in reply  # rule-layer matched condition, shown as-is
    assert NAMES["FUND_007"] in reply and "FUND_007" in reply and "Fund｜風險 High" in reply
    assert "本次已先產生客戶洞察" in reply and DISCLAIMER in reply


def test_rejected_recommendation_keeps_the_rule_result_and_hides_llm_text(paths):
    class BadWhat(FakeBackend):
        def generate(self, prompt, response_model):
            return NBALLMOutput(
                explanation="LLM 亂寫的說明",
                products=[ProductRecommendation(product_id="FUND_999", product_name="幻覺基金", reason="r")],
            )

    t = AgentTools(BadWhat(), fake_retriever, source_path=paths["all"], top10_path=paths["top10"],
                   insight_audit_log_path=paths["tmp"] / "i.jsonl", nba_audit_log_path=paths["tmp"] / "n.jsonl")
    result = t.run(TOOL_NBA, "C000104", SessionState())
    reply = render_response([result])
    assert result.nba.status == "rejected"
    assert "建議行動：A2" in reply and "未通過事實查核" in reply
    assert "LLM 亂寫的說明" not in reply and "幻覺基金" not in reply


def test_rejected_insight_shows_only_the_fallback_message(paths):
    bad = FakeBackend(insight_bad=True)
    t = AgentTools(bad, fake_retriever, source_path=paths["all"], top10_path=paths["top10"],
                   insight_audit_log_path=paths["tmp"] / "i.jsonl", nba_audit_log_path=paths["tmp"] / "n.jsonl")
    reply = render_response([t.run(TOOL_INSIGHT, "C000104", SessionState())])
    assert "未通過事實查核" in reply and "客戶投資意圖高" not in reply


def test_insight_reply_lists_the_validated_fields(tools):
    result = tools.run(TOOL_INSIGHT, "C000104", SessionState())
    reply = render_response([result])
    assert "客戶投資意圖高、尚未持有基金。" in reply and "risk_level = High" in reply and "通過事實查核" in reply


def test_summarize_result_never_contains_insight_text(tools):
    result = tools.run(TOOL_INSIGHT, "C000104", SessionState())
    assert "客戶投資意圖高" not in summarize_result(result)


def test_structured_prompt_lists_every_tool_and_the_question():
    context = build_context("今天要先聯繫誰？", "本次對話尚未討論任何客戶。", [])
    prompt = build_structured_prompt(context, [])
    assert TOOL_TOP10 in prompt and TOOL_INSIGHT in prompt and TOOL_NBA in prompt and "今天要先聯繫誰？" in prompt
