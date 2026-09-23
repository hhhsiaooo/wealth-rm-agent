"""Phase 4: tool layer for the Agent.

Three coarse-grained tools, each a thin wrapper over an already-validated
pipeline -- the Agent never re-implements business rules or validation:

- get_top_priority_customers()        WHO: today's precomputed priority list (no LLM)
- get_customer_insight(customer_id)   WHY: insight + fact validation + grounding policy
- get_next_best_action(customer_id)   WHAT: rules + retrieval + LLM + validation.
  If the conversation has no insight for that customer yet, WHY runs first
  (enforced here in code, not left to the LLM). If WHY is rejected, WHAT still
  runs without insight and the result says so.

Guardrails at this layer: tool-name allowlist, customer_id must exist in the
1,000-customer dataset before anything runs, and a failure inside a tool comes
back as an error result instead of crashing the conversation.

Every successful tool call also appends a per-stage latency record (see
agent/latency.py) -- data retrieval, WHY/WHAT LLM generation, RAG retrieval,
end-to-end -- to outputs/agent_latency_log.jsonl, for comparing against an
RM's own baseline time and for tracking later retrieval/prompt/model changes.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Protocol

import pandas as pd
from pydantic import BaseModel

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from agent.latency import LATENCY_LOG_PATH, LatencyRecord, TimingLLM, TimingRetriever, append_latency_log, now_iso  # noqa: E402
from insight.fact_validator import AUDIT_LOG_PATH, GroundingOutcome, generate_grounded_customer_insight  # noqa: E402
from insight.insight_generator import CustomerInsight  # noqa: E402
from rag.nba_generator import NBA_AUDIT_LOG_PATH, NBAOutcome, generate_grounded_next_best_action  # noqa: E402
from rag.product_knowledge import ProductKnowledge  # noqa: E402
from rag.recommendation_validator import default_catalog  # noqa: E402
from rag.retrieval import ProductRetriever  # noqa: E402
from tools.customer_tool import ALL_PATH, TOP10_PATH, CustomerDataError, get_customer_profile  # noqa: E402

TOOL_TOP10 = "get_top_priority_customers"
TOOL_INSIGHT = "get_customer_insight"
TOOL_NBA = "get_next_best_action"


class ToolSpec(BaseModel):
    name: str
    description: str
    needs_customer_id: bool


TOOL_SPECS = [
    ToolSpec(
        name=TOOL_TOP10,
        description="列出今日優先聯繫的客戶名單（排名、優先分數、三個子分數）。不需要客戶編號。",
        needs_customer_id=False,
    ),
    ToolSpec(
        name=TOOL_INSIGHT,
        description="說明指定客戶為什麼值得聯繫（客戶洞察）。適用 1,000 位客戶中的任何一位。",
        needs_customer_id=True,
    ),
    ToolSpec(
        name=TOOL_NBA,
        description="建議指定客戶的下一步行動與候選產品。若本次對話尚無該客戶的洞察，會自動先取得洞察。",
        needs_customer_id=True,
    ),
]
TOOL_NAMES = {spec.name for spec in TOOL_SPECS}
TOOLS_BY_NAME = {spec.name: spec for spec in TOOL_SPECS}


class BackendLLM(Protocol):
    """What the tools need from an LLM client: the WHY call (prompt ->
    CustomerInsight) and the WHAT call (structured generate). The Gemini and
    Anthropic clients in insight/insight_generator.py provide both."""

    def __call__(self, prompt: str) -> CustomerInsight: ...

    def generate(self, prompt: str, response_model: type) -> Any: ...


class SessionState(BaseModel):
    """In-memory state for one conversation. Insights are cached so a follow-up
    WHAT question does not pay for WHY again."""

    insights: dict[str, GroundingOutcome] = {}
    discussed: list[str] = []  # customer ids, oldest first

    def remember_customer(self, customer_id: str) -> None:
        if customer_id in self.discussed:
            self.discussed.remove(customer_id)
        self.discussed.append(customer_id)


class PriorityRow(BaseModel):
    priority_rank: int
    customer_id: str
    priority_score: float
    asset_potential_score: float
    investment_gap_score: float
    investment_intent_score: float


class ToolResult(BaseModel):
    tool: str
    customer_id: Optional[str] = None
    ok: bool
    error: Optional[str] = None
    top10: Optional[list[PriorityRow]] = None
    insight: Optional[GroundingOutcome] = None
    nba: Optional[NBAOutcome] = None
    # WHAT only: where the insight used as context came from.
    #   generated   = WHY was run just now because the session had none
    #   cached      = reused from earlier in this conversation
    #   unavailable = WHY was rejected, so WHAT ran without insight
    insight_source: Optional[Literal["generated", "cached", "unavailable"]] = None


def normalize_customer_id(raw: Optional[str]) -> str:
    """Only trim and upper-case: never guess or repair an id."""
    return (raw or "").strip().upper()


class LazyRetriever:
    """Builds the real retriever (Chroma + embeddings) on first use, so
    conversations that never need product retrieval do not load it."""

    def __init__(self, factory: Callable[[], ProductRetriever]) -> None:
        self._factory = factory
        self._retriever: Optional[ProductRetriever] = None

    def __call__(self, query: str, category: str, risk_levels=None) -> list[dict]:
        if self._retriever is None:
            self._retriever = self._factory()
        return self._retriever(query, category, risk_levels)


def default_retriever_factory() -> ProductRetriever:
    from rag.retrieval import GeminiEmbeddingClient, load_or_build_collection, make_retriever

    embedding_client = GeminiEmbeddingClient()
    return make_retriever(load_or_build_collection(embedding_client), embedding_client)


class AgentTools:
    def __init__(
        self,
        llm_client: BackendLLM,
        retriever: ProductRetriever,
        source_path: Path = ALL_PATH,
        top10_path: Path = TOP10_PATH,
        insight_audit_log_path: Path = AUDIT_LOG_PATH,
        nba_audit_log_path: Path = NBA_AUDIT_LOG_PATH,
        catalog: dict[str, ProductKnowledge] | None = None,
        latency_log_path: Path = LATENCY_LOG_PATH,
    ) -> None:
        self._llm = llm_client
        self._retriever = retriever
        self._source_path = source_path
        self._top10_path = top10_path
        self._insight_audit_log_path = insight_audit_log_path
        self._nba_audit_log_path = nba_audit_log_path
        self._catalog = catalog if catalog is not None else default_catalog()
        self._latency_log_path = latency_log_path

    # -- entry point -----------------------------------------------------------

    def run(self, tool_name: str, customer_id: Optional[str], session: SessionState) -> ToolResult:
        """Run one tool. Never raises: any failure becomes an error result."""
        spec = TOOLS_BY_NAME.get(tool_name)
        if spec is None:
            return ToolResult(tool=tool_name, ok=False, error=f"不支援的工具：{tool_name}")

        if not spec.needs_customer_id:
            return self._guarded(tool_name, None, lambda: self._top_priority(tool_name))

        cid = normalize_customer_id(customer_id)
        if not cid:
            return ToolResult(tool=tool_name, ok=False, error="缺少客戶編號，請提供要查詢的客戶編號（例如 C000104）。")
        t0 = time.perf_counter()
        try:
            get_customer_profile(cid, source_path=self._source_path)
        except CustomerDataError as exc:
            return ToolResult(tool=tool_name, customer_id=cid, ok=False, error=f"找不到客戶 {cid}：{exc}")
        except Exception as exc:  # noqa: BLE001 - reported to the user, not swallowed
            return ToolResult(tool=tool_name, customer_id=cid, ok=False, error=f"讀取客戶資料失敗：{type(exc).__name__}: {exc}")
        lookup_latency_sec = time.perf_counter() - t0

        session.remember_customer(cid)
        if tool_name == TOOL_INSIGHT:
            return self._guarded(tool_name, cid, lambda: self._customer_insight(cid, session, lookup_latency_sec))
        return self._guarded(tool_name, cid, lambda: self._next_best_action(cid, session, lookup_latency_sec))

    # -- tools -----------------------------------------------------------------

    def _guarded(self, tool_name: str, cid: Optional[str], fn: Callable[[], ToolResult]) -> ToolResult:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - LLM/API/data failures must not kill the conversation
            return ToolResult(
                tool=tool_name, customer_id=cid, ok=False,
                error=f"執行 {tool_name} 時發生錯誤：{type(exc).__name__}: {exc}",
            )

    def _log_latency(self, record: LatencyRecord) -> None:
        append_latency_log(record, self._latency_log_path)

    def _top_priority(self, tool_name: str) -> ToolResult:
        # WHO：只是讀已經算好的 CSV，完全不呼叫 LLM。
        t0 = time.perf_counter()
        df = pd.read_csv(self._top10_path).sort_values("priority_rank")
        rows = [
            PriorityRow(
                priority_rank=int(r["priority_rank"]),
                customer_id=str(r["customer_id"]),
                priority_score=float(r["priority_score"]),
                asset_potential_score=float(r["asset_potential_score"]),
                investment_gap_score=float(r["investment_gap_score"]),
                investment_intent_score=float(r["investment_intent_score"]),
            )
            for r in df.to_dict("records")
        ]
        self._log_latency(LatencyRecord(
            timestamp=now_iso(), tool=tool_name, end_to_end_latency_sec=time.perf_counter() - t0,
        ))
        return ToolResult(tool=tool_name, ok=True, top10=rows)

    def _customer_insight(self, cid: str, session: SessionState, lookup_latency_sec: float) -> ToolResult:
        # WHY：呼叫 generate_grounded_customer_insight()（insight/fact_validator.py），
        # 也就是「LLM 寫洞察 → 逐欄位比對原始資料 → 必要時重試 → 通過才顯示」整條流程。
        t0 = time.perf_counter()
        cached = session.insights.get(cid)
        why_llm_sec: Optional[float] = None
        why_attempts: Optional[int] = None
        # An explicit insight request retries a previously rejected insight;
        # a good one is reused so the same question does not cost another LLM call.
        if cached is None or cached.status == "rejected":
            timed_llm = TimingLLM(self._llm)
            cached = self._generate_insight(cid, session, llm_client=timed_llm)
            why_llm_sec, why_attempts = timed_llm.total_sec, cached.attempts
        self._log_latency(LatencyRecord(
            timestamp=now_iso(), tool=TOOL_INSIGHT, customer_id=cid,
            customer_lookup_latency_sec=lookup_latency_sec,
            why_llm_latency_sec=why_llm_sec, why_attempts=why_attempts,
            end_to_end_latency_sec=time.perf_counter() - t0,
        ))
        return ToolResult(tool=TOOL_INSIGHT, customer_id=cid, ok=True, insight=cached)

    def _next_best_action(self, cid: str, session: SessionState, lookup_latency_sec: float) -> ToolResult:
        t0 = time.perf_counter()
        cached = session.insights.get(cid)
        why_llm_sec: Optional[float] = None
        why_attempts: Optional[int] = None
        # 先確保有 WHY 洞察：對話裡已經問過就直接沿用（不再呼叫 LLM），
        # 沒問過就在這裡自動先跑一次 WHY，跑完才進入 WHAT。
        if cached is None:
            timed_why_llm = TimingLLM(self._llm)
            outcome = self._generate_insight(cid, session, llm_client=timed_why_llm)
            why_llm_sec, why_attempts = timed_why_llm.total_sec, outcome.attempts
            source = "generated"
        else:
            outcome, source = cached, "cached"
        if outcome.insight is None:  # WHY rejected (or not run and unavailable): WHAT still runs, without insight
            source = "unavailable"

        # WHAT：呼叫 generate_grounded_next_best_action()（rag/nba_generator.py），
        # 也就是「規則選行動 → RAG 找產品 → LLM 寫說明 → 查核 → 必要時重試」整條流程。
        timed_retriever = TimingRetriever(self._retriever)
        timed_what_llm = TimingLLM(self._llm)
        nba = generate_grounded_next_best_action(
            cid, timed_what_llm, timed_retriever,
            insight_text=outcome.insight.insight if outcome.insight else None,
            source_path=self._source_path,
            audit_log_path=self._nba_audit_log_path,
            catalog=self._catalog,
        )
        self._log_latency(LatencyRecord(
            timestamp=now_iso(), tool=TOOL_NBA, customer_id=cid,
            customer_lookup_latency_sec=lookup_latency_sec,
            why_llm_latency_sec=why_llm_sec, why_attempts=why_attempts,
            product_retrieval_latency_sec=timed_retriever.total_sec if timed_retriever.call_count else None,
            what_llm_latency_sec=timed_what_llm.total_sec, what_attempts=nba.attempts,
            end_to_end_latency_sec=time.perf_counter() - t0,
        ))
        return ToolResult(tool=TOOL_NBA, customer_id=cid, ok=True, nba=nba, insight_source=source)

    def _generate_insight(self, cid: str, session: SessionState, llm_client: Optional[BackendLLM] = None) -> GroundingOutcome:
        outcome = generate_grounded_customer_insight(
            cid, llm_client if llm_client is not None else self._llm,
            source_path=self._source_path, audit_log_path=self._insight_audit_log_path,
        )
        session.insights[cid] = outcome
        return outcome


__all__ = [
    "AgentTools", "BackendLLM", "LazyRetriever", "PriorityRow", "SessionState",
    "TOOL_INSIGHT", "TOOL_NAMES", "TOOL_NBA", "TOOL_SPECS", "TOOL_TOP10", "ToolResult",
    "default_retriever_factory", "normalize_customer_id",
]
