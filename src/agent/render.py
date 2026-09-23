"""Phase 4: code-rendered replies.

Everything the RM reads after a tool ran is produced here from the tools'
validated results. The LLM never writes this text, so it cannot add
unverified claims on top of the fact-checked insight / recommendation. (The
only LLM-written text shown is the insight / explanation / product reasons
that already passed validation, quoted as-is.)

簡單說：這個檔案不呼叫 LLM，也不會「生成」任何新句子。它只是把前面各階段
（規則層算的、LLM 寫且已通過查核的）結果，照固定版型組成理專看到的文字。
LLM 寫的洞察／說明／推薦理由會被原封不動引用進來，但排版本身是純程式邏輯。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from agent.agent_tools import TOOL_INSIGHT, TOOL_NBA, TOOL_TOP10, ToolResult  # noqa: E402
from rag.product_knowledge import ProductKnowledge  # noqa: E402
from rag.recommendation_validator import default_catalog  # noqa: E402

DISCLAIMER = "以上為決策輔助資訊，產品僅為候選，最終適合度與是否聯繫請理專自行判斷。"

INSIGHT_SOURCE_NOTES = {
    "generated": "本次已先產生客戶洞察，作為說明的參考。",
    "cached": "沿用本次對話已取得的客戶洞察。",
    "unavailable": "客戶洞察未通過事實查核，本次建議未使用洞察，僅依規則與產品資料產生。",
}


def render_top10(result: ToolResult) -> str:
    rows = result.top10 or []
    lines = [
        f"【今日優先名單】共 {len(rows)} 位（同分並列者皆列出）",
        "排名  客戶編號    優先分數  資產潛力  配置缺口  投資意圖",
    ]
    for r in rows:
        lines.append(
            f"{r.priority_rank:>4}  {r.customer_id:<10}  {r.priority_score:>7.1f}  "
            f"{r.asset_potential_score:>7.1f}  {r.investment_gap_score:>7.1f}  {r.investment_intent_score:>7.1f}"
        )
    return "\n".join(lines)


def render_insight(result: ToolResult) -> str:
    outcome = result.insight
    header = f"【客戶洞察】{result.customer_id}"
    if outcome is None or outcome.insight is None:
        message = outcome.fallback_message if outcome else "無法取得客戶洞察。"
        return f"{header}\n{message}"

    insight = outcome.insight
    status = "通過事實查核" if outcome.status == "grounded" else "經一次修正後通過事實查核"
    lines = [
        f"{header}（優先排名 {insight.priority_rank}，優先分數 {insight.priority_score:g}）",
        insight.insight,
        "依據欄位（皆已與客戶原始資料比對）：",
    ]
    lines += [f"  - {kf.field} = {kf.value}" for kf in insight.key_fields]
    lines.append(f"查核狀態：{status}")
    return "\n".join(lines)


def render_next_best_action(result: ToolResult, catalog: dict[str, ProductKnowledge]) -> str:
    outcome = result.nba
    header = f"【行動建議】{result.customer_id}"
    if outcome is None:
        return f"{header}\n無法取得行動建議。"

    selection = outcome.selection
    action_id = selection.recommended_action
    matched = next(a.matched_conditions for a in selection.evaluated_actions if a.action_id == action_id)
    initial = outcome.initial_next_best_action
    lines = [f"{header}　建議行動：{action_id} {initial.action_name}"]
    lines.append("成立條件：")
    lines += [f"  - {c}" for c in matched] or ["  - （無）"]

    nba = outcome.next_best_action
    if nba is None:  # explanation rejected: keep the rule result, hide the LLM text
        lines.append(outcome.fallback_message or "")
    else:
        lines.append(f"說明：{nba.explanation}")
        if nba.products:
            lines.append("候選產品（來自產品知識庫，仍需進一步確認適合度）：")
            for i, product in enumerate(nba.products, start=1):
                known = catalog.get(product.product_id)
                meta = (
                    f"{known.metadata.product_category}｜風險 {known.metadata.risk_level}｜{known.metadata.currency}"
                    if known else "資料庫查無此產品"
                )
                lines.append(f"  {i}. {product.product_name}（{product.product_id}｜{meta}）")
                lines.append(f"     {product.reason}")
    if result.insight_source:
        lines.append(INSIGHT_SOURCE_NOTES[result.insight_source])
    return "\n".join(line for line in lines if line is not None)


def render_error(result: ToolResult) -> str:
    return f"【無法完成】{result.error}"


def render_result(result: ToolResult, catalog: Optional[dict[str, ProductKnowledge]] = None) -> str:
    if not result.ok:
        return render_error(result)
    if result.tool == TOOL_TOP10:
        return render_top10(result)
    if result.tool == TOOL_INSIGHT:
        return render_insight(result)
    if result.tool == TOOL_NBA:
        return render_next_best_action(result, catalog if catalog is not None else default_catalog())
    return render_error(ToolResult(tool=result.tool, ok=False, error=f"沒有對應的顯示方式：{result.tool}"))


def render_response(results: list[ToolResult], catalog: Optional[dict[str, ProductKnowledge]] = None) -> str:
    """All results of one question, in call order, plus the disclaimer when
    any recommendation content was shown."""
    blocks = [render_result(r, catalog) for r in results]
    if any(r.ok for r in results):
        blocks.append(DISCLAIMER)
    return "\n\n".join(blocks)
