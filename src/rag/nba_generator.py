"""Phase 3.10 (WHAT): Next Best Action generation.

Flow (docs/ai_product_design.md, Task 3):
    rule layer (select_action_for_profile)     -- deterministic, no LLM
      -> Product Knowledge retrieval           -- only for A2/A3/A4
      -> LLM writes explanation + product picks (NBALLMOutput)
      -> code assembles NextBestAction (action/eligible/name/sources are
         copied/derived, never LLM-decided)
      -> validate_next_best_action() (3.11)
      -> Grounding Policy, same shape as WHY's: one corrective retry with the
         specific issues fed back, then reject.

On rejection the rule-layer result is still returned (it is deterministic and
trustworthy); only the LLM-written explanation is withheld.

Implementation choices, confirmed with the product owner (docs/ai_product_design.md):
- The search query is built deterministically from structured customer
  conditions (risk_level; A3 also the held deposit currency) + the action's
  category, instead of having the LLM turn Customer Evidence into a search
  need. Saves one LLM call per question and keeps the retrieval condition
  testable; the retriever is injectable, so an LLM-built query can replace
  build_search_query() later.
- The WHY insight text is used as context in the explanation prompt, not in
  the search query.
- Product risk vs. customer risk is a HARD metadata pre-filter (product risk
  must not exceed customer risk) -- a new suitability rule, not in the
  original spec, added at the product owner's request.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from insight.insight_generator import StructuredLLMClient  # noqa: E402
from rag.action_selector import select_action_for_profile  # noqa: E402
from rag.business_rules import (  # noqa: E402
    ACTION_NAMES,
    RETRIEVAL_CATEGORY_BY_ACTION,
    allowed_product_risk_levels,
)
from rag.product_knowledge import ProductKnowledge  # noqa: E402
from rag.recommendation_validator import SOURCE_PREFIX, default_catalog, validate_next_best_action  # noqa: E402
from rag.retrieval import ProductRetriever  # noqa: E402
from rag.schemas import (  # noqa: E402
    ActionSelectionResult,
    NBALLMOutput,
    NextBestAction,
    RecommendationCheckResult,
)
from tools.customer_tool import TOP10_PATH, get_customer_profile  # noqa: E402

NBA_AUDIT_LOG_PATH = Path(__file__).resolve(
).parents[2] / "outputs" / "nba_audit_log.jsonl"
FALLBACK_MESSAGE = "AI 行動建議說明未通過事實查核，僅提供規則判斷結果，請理專自行判斷。"

RISK_LABELS = {"Low": "低風險", "Medium": "中風險", "High": "高風險"}
CATEGORY_LABELS = {"Fund": "基金", "TD": "定存", "FX": "外幣"}
NEED_PHRASES = {
    "A2": "適合投資需求",
    "A3": "適合資金停泊與定存配置",
    "A4": "適合外幣資產配置",
}

NBA_RULES = """- 不可更改或質疑規則判斷結果，不可建立新的行動或業務規則。
- 不可捏造產品資訊，不可修改產品資料，不可推薦上述清單以外的產品。
- 不可做出最終適合度判斷；產品僅能描述為「可作為候選，仍需理專進一步確認適合度」。
- 不可執行交易或自動聯繫客戶。
- 客戶相關數值必須與上述資料完全一致，不可自行推算新的數字。"""


class NBAGenerationError(Exception):
    """Raised when the LLM client does not return an NBALLMOutput."""


class NBAContext(BaseModel):
    customer_id: str
    risk_level: str
    insight_text: Optional[str] = None
    selection: ActionSelectionResult
    search_query: Optional[str] = None
    query_evidence: list[str] = []
    allowed_risk_levels: list[str] = []
    hits: list[dict[str, Any]]


class NBAOutcome(BaseModel):
    customer_id: str
    status: Literal["grounded", "grounded_after_retry", "rejected"]
    selection: ActionSelectionResult
    next_best_action: Optional[NextBestAction] = None
    initial_next_best_action: NextBestAction
    initial_check: RecommendationCheckResult
    final_check: RecommendationCheckResult
    attempts: int
    fallback_message: Optional[str] = None


def build_query_evidence(profile: dict[str, Any], action_id: str) -> list[str]:
    """Structured customer conditions that shape the search text. Only
    conditions that map onto product attributes are used (risk level; for A3
    the held deposit currency). Everything else in the WHY insight explains
    why to contact now but does not discriminate between products, so it goes
    into the explanation prompt instead."""
    evidence = [RISK_LABELS.get(
        profile["risk_level"], profile["risk_level"]), NEED_PHRASES[action_id]]
    if action_id == "A3":
        has_twd, has_fx = profile["twd_td_balance"] > 0, profile["fx_td_balance"] > 0
        if has_twd and not has_fx:
            evidence.append("台幣")
        elif has_fx and not has_twd:
            evidence.append("外幣")
    return evidence


def build_search_query(evidence: list[str], action_id: str) -> str:
    category = RETRIEVAL_CATEGORY_BY_ACTION[action_id]
    return f"{'、'.join(evidence)}的{CATEGORY_LABELS[category]}產品"


def prepare_nba_context(
    customer_id: str,
    retriever: ProductRetriever,
    insight_text: Optional[str] = None,
    source_path: Path = TOP10_PATH,
) -> NBAContext:
    """Rule layer + (conditional) retrieval. Retrieval only runs when the
    selected action is A2/A3/A4."""
    profile = get_customer_profile(customer_id, source_path=source_path)
    # 規則層：純 Python if/else 判斷六個候選行動、選出唯一一個，完全不呼叫 LLM。
    selection = select_action_for_profile(profile)
    action_id = selection.recommended_action
    category = RETRIEVAL_CATEGORY_BY_ACTION.get(action_id)

    search_query: Optional[str] = None
    evidence: list[str] = []
    allowed: list[str] = []
    hits: list[dict[str, Any]] = []
    if category is not None:
        evidence = build_query_evidence(profile, action_id)
        search_query = build_search_query(evidence, action_id)
        allowed = allowed_product_risk_levels(profile["risk_level"])
        # RAG 檢索：用 Embedding 模型把查詢文字轉成向量做相似度比對找產品，
        # 這是「比對」不是「生成文字」，跟下面呼叫 LLM 寫說明是不同能力。
        hits = retriever(search_query, category, allowed)

    return NBAContext(
        customer_id=customer_id,
        risk_level=profile["risk_level"],
        insight_text=insight_text,
        selection=selection,
        search_query=search_query,
        query_evidence=evidence,
        allowed_risk_levels=allowed,
        hits=hits,
    )


def _format_product(product: ProductKnowledge) -> str:
    m, c = product.metadata, product.content
    return (
        f"- 產品代碼：{m.product_id}\n"
        f"  名稱：{c.product_name}\n"
        f"  類別／風險屬性／幣別：{m.product_category}／{m.risk_level}／{m.currency}\n"
        f"  產品特色：{c.product_features}\n"
        f"  適用客群：{c.target_customer}\n"
        f"  建議投資期間：{c.investment_horizon}\n"
        f"  注意事項：{c.sales_cautions}"
    )


def build_nba_prompt(ctx: NBAContext, catalog: dict[str, ProductKnowledge] | None = None) -> str:
    catalog = catalog if catalog is not None else default_catalog()
    action_id = ctx.selection.recommended_action
    matched = next(
        a.matched_conditions for a in ctx.selection.evaluated_actions if a.action_id == action_id)
    conditions = "\n".join(f"- {c}" for c in matched) or "- （無）"
    insight = ctx.insight_text or "（本次沒有可用的客戶洞察，請僅依規則判斷結果與客戶條件撰寫。）"

    if ctx.hits:
        products = "\n".join(_format_product(
            catalog[h["product_id"]]) for h in ctx.hits if h["product_id"] in catalog)
        product_section = f"【檢索到的產品資料】\n{products}"
        product_task = (
            "2. products：從上述檢索到的產品中挑 1～3 個最相關的。每個包含 product_id、"
            "product_name（必須與產品資料完全相同）、reason（依據產品資料說明可作為候選的原因，"
            "並如實陳述產品風險屬性與客戶風險屬性的關係，僅陳述事實，不作適合度判斷）。"
        )
    else:
        product_section = "【產品資料】此行動不需要產品檢索。"
        product_task = "2. products：必須是空陣列。"

    return f"""你是財富管理理專的行動建議助理。以下資訊都已經過程式驗證，請只根據這些資訊撰寫建議行動說明。

【規則判斷結果（由程式決定，你不可更改）】
客戶：{ctx.customer_id}
建議行動：{action_id} {ACTION_NAMES[action_id]}
成立條件：
{conditions}

【客戶條件】
風險屬性：{ctx.risk_level}

【客戶洞察（已通過事實查核）】
{insight}

{product_section}

請你：
1. explanation：用繁體中文寫一段簡潔的說明，解釋為什麼建議此行動（可引用成立條件與客戶洞察）。
{product_task}

規則（必須遵守）：
{NBA_RULES}
"""


def build_nba_correction_prompt(
    ctx: NBAContext,
    previous: NBALLMOutput,
    issues: list[str],
    catalog: dict[str, ProductKnowledge] | None = None,
) -> str:
    issues_text = "\n".join(f"- {i}" for i in issues)
    return (
        build_nba_prompt(ctx, catalog)
        + f"\n【先前輸出未通過查核】\n先前的輸出：\n{previous.model_dump_json()}\n"
        f"發現的問題：\n{issues_text}\n請修正上述所有問題後重新輸出。\n"
    )


def assemble_next_best_action(ctx: NBAContext, output: NBALLMOutput) -> NextBestAction:
    """recommended_action / eligible / action_name are copied from the rule
    layer; sources are derived from the picked products. The LLM controls
    only explanation and product picks."""
    recommended = ctx.selection.recommended_action
    eligible = next(
        a.eligible for a in ctx.selection.evaluated_actions if a.action_id == recommended)
    sources = list(dict.fromkeys(
        f"{SOURCE_PREFIX}{p.product_id}" for p in output.products))
    return NextBestAction(
        customer_id=ctx.customer_id,
        recommended_action=recommended,
        action_name=ACTION_NAMES[recommended],
        eligible=eligible,
        explanation=output.explanation,
        products=output.products,
        sources=sources,
    )


def _generate(llm_client: StructuredLLMClient, prompt: str) -> NBALLMOutput:
    result = llm_client.generate(prompt, NBALLMOutput)
    if not isinstance(result, NBALLMOutput):
        raise NBAGenerationError(
            f"llm_client returned {type(result).__name__}, expected NBALLMOutput")
    return result


def _append_audit_log(record: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_grounded_next_best_action(
    customer_id: str,
    llm_client: StructuredLLMClient,
    retriever: ProductRetriever,
    insight_text: Optional[str] = None,
    source_path: Path = TOP10_PATH,
    audit_log_path: Path = NBA_AUDIT_LOG_PATH,
    catalog: dict[str, ProductKnowledge] | None = None,
) -> NBAOutcome:
    """Rule layer -> retrieval -> LLM -> validate -> (corrective retry once)
    -> reject. Every attempt is appended to audit_log_path.

    Raises:
        CustomerNotFoundError / CustomerSchemaError / CustomerDataError:
            propagated from get_customer_profile().
        NBAGenerationError: if llm_client returns something that is not an
            NBALLMOutput.
    """
    catalog = catalog if catalog is not None else default_catalog()
    def now(): return datetime.now(timezone.utc).isoformat()  # noqa: E731

    ctx = prepare_nba_context(customer_id, retriever,
                              insight_text, source_path)

    # 真正呼叫 LLM 的地方：它只看得到、只能寫 explanation 和 products 兩個欄位
    # （NBALLMOutput），action/eligible 等其他欄位它看不到也改不了。
    output = _generate(llm_client, build_nba_prompt(ctx, catalog))
    # 組裝：把規則層算好的 action/eligible/name 跟 LLM 寫的 explanation/products
    # 合成一個 NextBestAction，兩種來源的欄位混在同一個物件裡。
    initial_nba = assemble_next_best_action(ctx, output)
    # 查核：確認 LLM 沒有推薦檢索結果以外的產品、沒有改產品名稱、風險沒超標。
    initial_check = validate_next_best_action(
        initial_nba, ctx.selection, ctx.hits, catalog, ctx.risk_level)
    _append_audit_log({
        "timestamp": now(), "customer_id": customer_id, "attempt": 1,
        "recommended_action": ctx.selection.recommended_action,
        "is_grounded": initial_check.is_grounded, "issues": initial_check.issues,
    }, audit_log_path)

    if initial_check.is_grounded:
        outcome = NBAOutcome(
            customer_id=customer_id, status="grounded", selection=ctx.selection,
            next_best_action=initial_nba, initial_next_best_action=initial_nba,
            initial_check=initial_check, final_check=initial_check, attempts=1,
        )
        _append_audit_log({"timestamp": now(), "customer_id": customer_id,
                          "final_status": outcome.status}, audit_log_path)
        return outcome

    retry_output = _generate(llm_client, build_nba_correction_prompt(
        ctx, output, initial_check.issues, catalog))
    retry_nba = assemble_next_best_action(ctx, retry_output)
    final_check = validate_next_best_action(
        retry_nba, ctx.selection, ctx.hits, catalog, ctx.risk_level)
    _append_audit_log({
        "timestamp": now(), "customer_id": customer_id, "attempt": 2,
        "recommended_action": ctx.selection.recommended_action,
        "is_grounded": final_check.is_grounded, "issues": final_check.issues,
    }, audit_log_path)

    if final_check.is_grounded:
        outcome = NBAOutcome(
            customer_id=customer_id, status="grounded_after_retry", selection=ctx.selection,
            next_best_action=retry_nba, initial_next_best_action=initial_nba,
            initial_check=initial_check, final_check=final_check, attempts=2,
        )
    else:
        outcome = NBAOutcome(
            customer_id=customer_id, status="rejected", selection=ctx.selection,
            next_best_action=None, initial_next_best_action=initial_nba,
            initial_check=initial_check, final_check=final_check, attempts=2,
            fallback_message=FALLBACK_MESSAGE,
        )
    _append_audit_log({"timestamp": now(), "customer_id": customer_id,
                      "final_status": outcome.status}, audit_log_path)
    return outcome
