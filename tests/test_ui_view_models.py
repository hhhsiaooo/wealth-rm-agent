from __future__ import annotations

# Reuse the Agent tests' fakes and fixtures (paths / backend / tools).
from test_agent import FakeBackend, backend, fake_retriever, paths, tools  # noqa: F401
from test_agent import make_row  # noqa: F401

from agent.agent_tools import TOOL_INSIGHT, TOOL_NBA, TOOL_TOP10, SessionState, ToolResult
from agent.rm_agent import AgentResponse
from agent.agent_tools import AgentTools
from ui.view_models import (
    ErrorBlock,
    InsightBlock,
    RecommendationBlock,
    TextBlock,
    TopListBlock,
    block_from_result,
    blocks_from_response,
    insight_status_badge,
    recommendation_status_badge,
    shows_disclaimer,
    top_list_rows,
)


def test_status_badges_map_every_status_to_a_tone():
    assert insight_status_badge("grounded").tone == "success"
    assert insight_status_badge("grounded_after_retry").tone == "warning"
    assert insight_status_badge("rejected").tone == "error"
    assert recommendation_status_badge("grounded").tone == "success"
    assert recommendation_status_badge("grounded_after_retry").tone == "warning"
    assert recommendation_status_badge("rejected").tone == "error"
    assert "修正" in insight_status_badge("grounded_after_retry").label


def test_top_list_block_has_readable_columns(tools):
    result = tools.run(TOOL_TOP10, None, SessionState())
    block = block_from_result(result)
    assert isinstance(block, TopListBlock)
    assert block.rows[0] == {
        "排名": 1, "客戶編號": "C000104", "優先分數": 90.5,
        "資產潛力": 80.0, "配置缺口": 94.0, "投資意圖": 92.0,
    }
    assert [r["客戶編號"] for r in top_list_rows(result)] == ["C000104", "C000518"]


def test_grounded_insight_block_carries_text_fields_and_a_success_badge(tools):
    block = block_from_result(tools.run(TOOL_INSIGHT, "C000104", SessionState()))
    assert isinstance(block, InsightBlock)
    assert block.text == "客戶投資意圖高、尚未持有基金。" and block.status.tone == "success"
    assert (block.priority_rank, block.priority_score) == (1, 90.5)
    assert [(f.field, f.value) for f in block.key_fields] == [("risk_level", "High")]  # values shown as text


def test_rejected_insight_block_shows_no_ai_text(paths):
    bad = FakeBackend(insight_bad=True)
    t = AgentTools(bad, fake_retriever, source_path=paths["all"], top10_path=paths["top10"],
                   insight_audit_log_path=paths["tmp"] / "i.jsonl", nba_audit_log_path=paths["tmp"] / "n.jsonl")
    block = block_from_result(t.run(TOOL_INSIGHT, "C000104", SessionState()))
    assert isinstance(block, InsightBlock)
    assert block.text is None and block.key_fields == []
    assert block.status.tone == "error" and "未通過事實查核" in block.message


def test_recommendation_block_has_rule_result_product_cards_and_insight_note(tools):
    block = block_from_result(tools.run(TOOL_NBA, "C000104", SessionState()))
    assert isinstance(block, RecommendationBlock)
    assert (block.action_id, block.action_name) == ("A2", "基金產品介紹")
    assert "investment_intent_score=92.0 >= 70" in block.conditions
    assert block.explanation == "建議介紹基金。" and block.status.tone == "success"
    assert [(p.product_id, p.meta) for p in block.products] == [("FUND_007", "Fund｜風險 High｜TWD")]
    assert "已先產生客戶洞察" in block.insight_note


def test_rejected_recommendation_keeps_rule_result_and_hides_llm_text(paths):
    from rag.schemas import NBALLMOutput, ProductRecommendation

    class BadWhat(FakeBackend):
        def generate(self, prompt, response_model):
            return NBALLMOutput(explanation="LLM 亂寫的說明", products=[
                ProductRecommendation(product_id="FUND_999", product_name="幻覺基金", reason="r")])

    t = AgentTools(BadWhat(), fake_retriever, source_path=paths["all"], top10_path=paths["top10"],
                   insight_audit_log_path=paths["tmp"] / "i.jsonl", nba_audit_log_path=paths["tmp"] / "n.jsonl")
    block = block_from_result(t.run(TOOL_NBA, "C000104", SessionState()))
    assert isinstance(block, RecommendationBlock)
    assert block.action_id == "A2" and block.conditions  # rule layer kept
    assert block.explanation is None and block.products == []  # nothing the LLM wrote
    assert block.status.tone == "error" and "未通過" in block.message


def test_failed_tool_becomes_an_error_block(tools):
    block = block_from_result(tools.run(TOOL_INSIGHT, "C999999", SessionState()))
    assert isinstance(block, ErrorBlock) and "C999999" in block.message


def test_unknown_tool_result_becomes_an_error_block():
    block = block_from_result(ToolResult(tool="mystery", ok=True))
    assert isinstance(block, ErrorBlock)


def test_response_without_tool_results_is_a_text_block():
    blocks = blocks_from_response(AgentResponse(reply="請提供客戶編號。"))
    assert blocks == [TextBlock(text="請提供客戶編號。")]


def test_response_with_tool_results_has_one_block_per_result_in_order(tools):
    session = SessionState()
    results = [tools.run(TOOL_TOP10, None, session), tools.run(TOOL_NBA, "C000104", session)]
    blocks = blocks_from_response(AgentResponse(reply="ignored", tool_results=results))
    assert [b.kind for b in blocks] == ["top_list", "recommendation"]


def test_disclaimer_only_when_recommendation_content_is_shown(tools):
    ok_blocks = [block_from_result(tools.run(TOOL_TOP10, None, SessionState()))]
    assert shows_disclaimer(ok_blocks)
    assert not shows_disclaimer([ErrorBlock(message="x"), TextBlock(text="y")])
