"""Phase 5 (UI): turns validated tool results into display blocks.

Pure functions, no Streamlit -- so what the RM sees can be tested without a
browser. The blocks are built only from the tools' validated results (the same
source render.py uses for the terminal): the UI never shows free LLM text
other than the insight / explanation / product reasons that already passed
validation, and it makes the guardrails visible (check status badges,
evidence fields, what the Agent did).

簡單說：Agent 跑完一個問題後拿到的是結構化資料（ToolResult），不是文字。
這個檔案把那份資料轉成一個個「畫面積木」（下面的 XxxBlock），例如
TopListBlock 是名單表格、InsightBlock 是一則洞察、RecommendationBlock 是
一則行動建議。app.py 再照這些積木畫成畫面。這裡完全不呼叫 LLM，只是把已經
產生好、已經查核過的結果換個形狀。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from agent.agent_tools import TOOL_INSIGHT, TOOL_NBA, TOOL_TOP10, ToolResult  # noqa: E402
from agent.render import DISCLAIMER, INSIGHT_SOURCE_NOTES  # noqa: E402
from agent.rm_agent import AgentResponse  # noqa: E402
from rag.product_knowledge import ProductKnowledge  # noqa: E402
from rag.recommendation_validator import default_catalog  # noqa: E402

Tone = Literal["success", "warning", "error"]

TOP_LIST_COLUMNS = ["排名", "客戶編號", "優先分數", "資產潛力", "配置缺口", "投資意圖"]


class StatusBadge(BaseModel):
    tone: Tone
    label: str


class FieldRow(BaseModel):
    field: str
    value: str


class ProductCard(BaseModel):
    name: str
    product_id: str
    meta: str
    reason: str


class TopListBlock(BaseModel):
    kind: Literal["top_list"] = "top_list"
    rows: list[dict[str, Any]]


class InsightBlock(BaseModel):
    kind: Literal["insight"] = "insight"
    customer_id: str
    priority_rank: Optional[int] = None
    priority_score: Optional[float] = None
    text: Optional[str] = None  # None when the insight was rejected: nothing AI-written is shown
    key_fields: list[FieldRow] = []
    status: StatusBadge
    message: Optional[str] = None  # fixed fallback text when rejected


class RecommendationBlock(BaseModel):
    kind: Literal["recommendation"] = "recommendation"
    customer_id: str
    action_id: str
    action_name: str
    conditions: list[str]
    explanation: Optional[str] = None  # None when the explanation was rejected
    products: list[ProductCard] = []
    status: StatusBadge
    insight_note: Optional[str] = None
    message: Optional[str] = None  # fixed fallback text when rejected


class ErrorBlock(BaseModel):
    kind: Literal["error"] = "error"
    message: str


class TextBlock(BaseModel):
    kind: Literal["text"] = "text"
    text: str


Block = Union[TopListBlock, InsightBlock, RecommendationBlock, ErrorBlock, TextBlock]


def insight_status_badge(status: str) -> StatusBadge:
    if status == "grounded":
        return StatusBadge(tone="success", label="通過事實查核")
    if status == "grounded_after_retry":
        return StatusBadge(tone="warning", label="經一次修正後通過事實查核")
    return StatusBadge(tone="error", label="未通過事實查核，不顯示 AI 洞察")


def recommendation_status_badge(status: str) -> StatusBadge:
    if status == "grounded":
        return StatusBadge(tone="success", label="說明通過推薦查核")
    if status == "grounded_after_retry":
        return StatusBadge(tone="warning", label="說明經一次修正後通過推薦查核")
    return StatusBadge(tone="error", label="說明未通過推薦查核，僅顯示規則判斷結果")


def top_list_rows(result: ToolResult) -> list[dict[str, Any]]:
    return [
        {
            "排名": r.priority_rank, "客戶編號": r.customer_id, "優先分數": r.priority_score,
            "資產潛力": r.asset_potential_score, "配置缺口": r.investment_gap_score,
            "投資意圖": r.investment_intent_score,
        }
        for r in result.top10 or []
    ]


def _insight_block(result: ToolResult) -> InsightBlock:
    outcome = result.insight
    status = insight_status_badge(outcome.status if outcome else "rejected")
    if outcome is None or outcome.insight is None:
        return InsightBlock(
            customer_id=result.customer_id or "", status=status,
            message=(outcome.fallback_message if outcome else None) or "無法取得客戶洞察。",
        )
    insight = outcome.insight
    return InsightBlock(
        customer_id=result.customer_id or insight.customer_id,
        priority_rank=insight.priority_rank, priority_score=insight.priority_score,
        text=insight.insight,
        key_fields=[FieldRow(field=kf.field, value=str(kf.value)) for kf in insight.key_fields],
        status=status,
    )


def _recommendation_block(result: ToolResult, catalog: dict[str, ProductKnowledge]) -> Union[RecommendationBlock, ErrorBlock]:
    outcome = result.nba
    if outcome is None:
        return ErrorBlock(message="無法取得行動建議。")
    selection = outcome.selection
    action_id = selection.recommended_action
    conditions = next(a.matched_conditions for a in selection.evaluated_actions if a.action_id == action_id)
    nba = outcome.next_best_action

    products: list[ProductCard] = []
    if nba is not None:
        for product in nba.products:
            known = catalog.get(product.product_id)
            meta = (
                f"{known.metadata.product_category}｜風險 {known.metadata.risk_level}｜{known.metadata.currency}"
                if known else "資料庫查無此產品"
            )
            products.append(ProductCard(name=product.product_name, product_id=product.product_id, meta=meta, reason=product.reason))

    return RecommendationBlock(
        customer_id=result.customer_id or "",
        action_id=action_id,
        action_name=outcome.initial_next_best_action.action_name,
        conditions=list(conditions),
        explanation=nba.explanation if nba is not None else None,
        products=products,
        status=recommendation_status_badge(outcome.status),
        insight_note=INSIGHT_SOURCE_NOTES.get(result.insight_source or ""),
        message=outcome.fallback_message if nba is None else None,
    )


def block_from_result(result: ToolResult, catalog: Optional[dict[str, ProductKnowledge]] = None) -> Block:
    if not result.ok:
        return ErrorBlock(message=result.error or "發生未知錯誤。")
    if result.tool == TOOL_TOP10:
        return TopListBlock(rows=top_list_rows(result))
    if result.tool == TOOL_INSIGHT:
        return _insight_block(result)
    if result.tool == TOOL_NBA:
        return _recommendation_block(result, catalog if catalog is not None else default_catalog())
    return ErrorBlock(message=f"沒有對應的顯示方式：{result.tool}")


def blocks_from_response(response: AgentResponse, catalog: Optional[dict[str, ProductKnowledge]] = None) -> list[Block]:
    """No tool ran -> the reply is the LLM's clarification / "unsupported" message
    (or an error notice). Otherwise one block per tool result, in call order."""
    if not response.tool_results:
        return [TextBlock(text=response.reply)]
    return [block_from_result(r, catalog) for r in response.tool_results]


def shows_disclaimer(blocks: list[Block]) -> bool:
    return any(not isinstance(b, (ErrorBlock, TextBlock)) for b in blocks)


__all__ = [
    "Block", "DISCLAIMER", "ErrorBlock", "FieldRow", "InsightBlock", "ProductCard", "RecommendationBlock",
    "StatusBadge", "TOP_LIST_COLUMNS", "TextBlock", "TopListBlock", "block_from_result",
    "blocks_from_response", "insight_status_badge", "recommendation_status_badge", "shows_disclaimer",
    "top_list_rows",
]
