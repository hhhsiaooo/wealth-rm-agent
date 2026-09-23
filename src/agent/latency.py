"""Phase 4/5: per-stage latency logging for the Agent's real usage.

簡單說：每次工具呼叫成功後，把「取資料花多久、WHY LLM 花多久、RAG 檢索花
多久、WHAT LLM 花多久、全部加起來花多久」各自量出來寫一行到 log，之後可以
用來跟理專原本作業時間比較，也可以在改動檢索或模型後比對有沒有變快。

CLAUDE.md's "System Performance" section asks these to be recorded
separately: data retrieval latency, product retrieval latency, LLM
generation latency, end-to-end latency. evaluation/why_evaluation.py and
evaluation/what_evaluation.py already measure these, but only for a fixed
evaluation set run manually (a paid, one-off command). This module measures
the SAME stages for every real question asked through the Agent (CLI
rm_agent.py / UI ui/app.py), appended to outputs/agent_latency_log.jsonl --
so a real question's timing can be compared against an RM's own baseline,
and later against retrieval/prompt/model changes (Phase 5 experiments),
without re-running a paid evaluation.

A stage field is None, never 0.0, when that stage did not run for this call
(e.g. why_llm_latency_sec is None when the insight was served from the
session cache, or when the tool is get_top_priority_customers) -- the same
"never fabricate a number" principle used elsewhere in this project (see
evaluation/what_evaluation.py's Rate).

Known limitation: only successful tool calls are logged. A failed call's
stage timings are incomplete (the exception aborts partway); correctness of
every attempt, success or failure, is already covered by the separate audit
logs (grounding_audit_log.jsonl / nba_audit_log.jsonl) -- this log is about
performance, not correctness, and the two are kept separate on purpose.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

LATENCY_LOG_PATH = Path(__file__).resolve().parents[2] / "outputs" / "agent_latency_log.jsonl"


class LatencyRecord(BaseModel):
    timestamp: str
    tool: str  # get_top_priority_customers / get_customer_insight / get_next_best_action
    customer_id: Optional[str] = None
    # get_customer_profile's guardrail lookup in AgentTools.run() -- a proxy
    # for "data retrieval latency": the pipelines re-read the same record
    # internally, so this is representative, not the full cost of every read.
    customer_lookup_latency_sec: Optional[float] = None
    why_llm_latency_sec: Optional[float] = None  # WHY LLM call(s) this tool call actually made
    why_attempts: Optional[int] = None  # 1 or 2 (Grounding Policy's one retry); None if WHY did not run
    product_retrieval_latency_sec: Optional[float] = None  # None when the action needs no retrieval (A1/A5/A6)
    what_llm_latency_sec: Optional[float] = None  # None when the tool is not get_next_best_action
    what_attempts: Optional[int] = None
    end_to_end_latency_sec: float  # full tool call, wall time


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_latency_log(record: LatencyRecord, path: Path = LATENCY_LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")


class TimingLLM:
    """Wraps a BackendLLM (WHY's __call__, WHAT's .generate) to measure pure
    LLM time across however many calls one tool call makes (the Grounding
    Policy's one corrective retry). Instantiate one per tool call -- never
    share an instance across calls, or the totals would mix stages."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.total_sec = 0.0

    def __call__(self, prompt: str) -> Any:
        t0 = time.perf_counter()
        try:
            return self._inner(prompt)
        finally:
            self.total_sec += time.perf_counter() - t0

    def generate(self, prompt: str, response_model: type) -> Any:
        t0 = time.perf_counter()
        try:
            return self._inner.generate(prompt, response_model)
        finally:
            self.total_sec += time.perf_counter() - t0


class TimingRetriever:
    """Wraps a ProductRetriever to measure total retrieval time for one tool
    call (currently at most one retrieval call per WHAT question). Also keeps
    the hits of the most recent call, since some callers (WHAT evaluation)
    need to see what was retrieved but the generation pipeline doesn't
    expose it."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.total_sec = 0.0
        self.call_count = 0
        self.last_hits: list[dict] = []

    def __call__(self, query: str, category: str, risk_levels: Any = None) -> list[dict]:
        t0 = time.perf_counter()
        try:
            self.last_hits = self._inner(query, category, risk_levels)
            return self.last_hits
        finally:
            self.total_sec += time.perf_counter() - t0
            self.call_count += 1


__all__ = [
    "LATENCY_LOG_PATH", "LatencyRecord", "TimingLLM", "TimingRetriever",
    "append_latency_log", "now_iso",
]
