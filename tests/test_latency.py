from __future__ import annotations

import json
import time

import pandas as pd
import pytest

from agent.agent_tools import TOOL_INSIGHT, TOOL_NBA, TOOL_TOP10, AgentTools, SessionState
from agent.latency import LatencyRecord, TimingLLM, TimingRetriever, append_latency_log
from rag.schemas import NBALLMOutput, ProductRecommendation
from test_agent import FakeBackend, fake_retriever, make_row
from tools.customer_tool import EXPECTED_FIELDS


@pytest.fixture
def paths(tmp_path):
    rows = [
        make_row("C000104", 1, investment_intent_score=92.0),            # A2: needs retrieval
        make_row("C000301", 3, risk_level="Low"),                        # A6: no retrieval
    ]
    all_path, top10_path = tmp_path / "all.csv", tmp_path / "top10.csv"
    df = pd.DataFrame(rows, columns=EXPECTED_FIELDS)
    df.to_csv(all_path, index=False)
    df.to_csv(top10_path, index=False)
    return {"all": all_path, "top10": top10_path, "tmp": tmp_path}


def make_tools(paths, backend, latency_log=None) -> AgentTools:
    return AgentTools(
        backend, fake_retriever, source_path=paths["all"], top10_path=paths["top10"],
        insight_audit_log_path=paths["tmp"] / "i.jsonl", nba_audit_log_path=paths["tmp"] / "n.jsonl",
        latency_log_path=latency_log or (paths["tmp"] / "latency.jsonl"),
    )


def read_log(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# -- TimingLLM / TimingRetriever (pure wrappers, no I/O) -------------------------

def test_timing_llm_accumulates_across_repeated_calls():
    class SlowLLM:
        def __call__(self, prompt):
            time.sleep(0.01)
            return "insight"

        def generate(self, prompt, response_model):
            time.sleep(0.01)
            return "nba"

    timed = TimingLLM(SlowLLM())
    assert timed("p") == "insight"
    assert timed.generate("p", object) == "nba"
    # Loose bound (not 2x0.01s): the point is "both calls accumulated, not just
    # the last one overwriting the total" -- a tight bound is flaky under load
    # (thread scheduling can delay a 0.01s sleep's wake-up).
    assert timed.total_sec >= 0.015


def test_timing_retriever_counts_calls_and_time():
    timed = TimingRetriever(lambda q, c, r=None: [{"product_id": "FUND_007"}])
    timed("q", "Fund", ["High"])
    assert timed.call_count == 1 and timed.total_sec >= 0


def test_timing_wrappers_still_time_when_the_inner_call_raises():
    def boom(prompt):
        raise RuntimeError("503")

    timed = TimingLLM(boom)
    with pytest.raises(RuntimeError):
        timed("p")
    assert timed.total_sec >= 0  # elapsed time recorded even though the call failed


# -- append_latency_log -----------------------------------------------------------

def test_append_latency_log_writes_one_json_line_per_call(tmp_path):
    path = tmp_path / "latency.jsonl"
    append_latency_log(LatencyRecord(timestamp="t1", tool=TOOL_TOP10, end_to_end_latency_sec=0.01), path)
    append_latency_log(LatencyRecord(timestamp="t2", tool=TOOL_TOP10, end_to_end_latency_sec=0.02), path)
    rows = read_log(path)
    assert len(rows) == 2 and rows[0]["timestamp"] == "t1" and rows[1]["end_to_end_latency_sec"] == 0.02


def test_stage_not_run_is_null_not_zero(tmp_path):
    path = tmp_path / "latency.jsonl"
    append_latency_log(LatencyRecord(timestamp="t", tool=TOOL_INSIGHT, end_to_end_latency_sec=0.01), path)
    row = read_log(path)[0]
    assert row["why_llm_latency_sec"] is None and row["product_retrieval_latency_sec"] is None


# -- wired into AgentTools ---------------------------------------------------------

def test_top10_call_logs_end_to_end_only(paths):
    tools = make_tools(paths, FakeBackend())
    tools.run(TOOL_TOP10, None, SessionState())
    row = read_log(paths["tmp"] / "latency.jsonl")[0]
    assert row["tool"] == TOOL_TOP10 and row["customer_id"] is None
    assert row["end_to_end_latency_sec"] >= 0
    assert row["why_llm_latency_sec"] is None and row["what_llm_latency_sec"] is None


def test_insight_call_logs_lookup_and_why_llm_time_with_attempts(paths):
    backend = FakeBackend()
    tools = make_tools(paths, backend)
    tools.run(TOOL_INSIGHT, "C000104", SessionState())
    row = read_log(paths["tmp"] / "latency.jsonl")[0]
    assert row["tool"] == TOOL_INSIGHT and row["customer_id"] == "C000104"
    assert row["customer_lookup_latency_sec"] >= 0
    assert row["why_llm_latency_sec"] >= 0 and row["why_attempts"] == 1
    assert row["end_to_end_latency_sec"] >= row["why_llm_latency_sec"]
    assert row["product_retrieval_latency_sec"] is None and row["what_llm_latency_sec"] is None


def test_cached_insight_reuse_logs_no_why_llm_time(paths):
    backend = FakeBackend()
    tools = make_tools(paths, backend)
    session = SessionState()
    tools.run(TOOL_INSIGHT, "C000104", session)
    tools.run(TOOL_INSIGHT, "C000104", session)  # second ask: served from session cache
    rows = read_log(paths["tmp"] / "latency.jsonl")
    assert len(rows) == 2
    assert rows[0]["why_llm_latency_sec"] is not None
    assert rows[1]["why_llm_latency_sec"] is None and rows[1]["why_attempts"] is None


def test_nba_with_retrieval_logs_why_retrieval_and_what_stages(paths):
    backend = FakeBackend()
    tools = make_tools(paths, backend)
    tools.run(TOOL_NBA, "C000104", SessionState())  # A2: no cached insight -> WHY runs, then retrieval
    row = read_log(paths["tmp"] / "latency.jsonl")[0]
    assert row["tool"] == TOOL_NBA
    assert row["why_llm_latency_sec"] is not None and row["why_attempts"] == 1
    assert row["product_retrieval_latency_sec"] is not None
    assert row["what_llm_latency_sec"] is not None and row["what_attempts"] == 1
    assert row["end_to_end_latency_sec"] >= row["why_llm_latency_sec"] + row["product_retrieval_latency_sec"]


def test_nba_reusing_a_cached_insight_logs_no_why_stage(paths):
    backend = FakeBackend()
    tools = make_tools(paths, backend)
    session = SessionState()
    tools.run(TOOL_INSIGHT, "C000104", session)
    tools.run(TOOL_NBA, "C000104", session)
    rows = read_log(paths["tmp"] / "latency.jsonl")
    nba_row = rows[1]
    assert nba_row["why_llm_latency_sec"] is None and nba_row["why_attempts"] is None
    assert nba_row["what_llm_latency_sec"] is not None


def test_nba_for_a_non_retrieval_action_logs_no_retrieval_stage(paths):
    backend = FakeBackend(rank_by_id={"C000301": 3})
    tools = make_tools(paths, backend)
    result = tools.run(TOOL_NBA, "C000301", SessionState())  # A6: eligible for nothing -> no retrieval
    assert result.nba.selection.recommended_action == "A6"
    row = read_log(paths["tmp"] / "latency.jsonl")[0]
    assert row["product_retrieval_latency_sec"] is None
    assert row["what_llm_latency_sec"] is not None  # the LLM still writes an explanation for A6


def test_why_generation_time_reflects_a_grounding_retry(paths):
    class RetryOnceBackend(FakeBackend):
        def __init__(self):
            super().__init__()
            self.insight_calls = 0

        def __call__(self, prompt):
            self.insight_calls += 1
            self.insight_bad = self.insight_calls == 1  # first attempt fails validation, second succeeds
            return super().__call__(prompt)

    tools = make_tools(paths, RetryOnceBackend())
    tools.run(TOOL_INSIGHT, "C000104", SessionState())
    row = read_log(paths["tmp"] / "latency.jsonl")[0]
    assert row["why_attempts"] == 2  # both LLM calls' time is included


def test_failed_tool_call_is_not_logged(paths):
    tools = make_tools(paths, FakeBackend())
    tools.run(TOOL_INSIGHT, "C999999", SessionState())  # unknown customer: fails before any tool logic
    assert not (paths["tmp"] / "latency.jsonl").exists()


def test_rejected_recommendation_still_logs_what_llm_time(paths):
    class BadWhat(FakeBackend):
        def generate(self, prompt, response_model):
            return NBALLMOutput(explanation="x", products=[
                ProductRecommendation(product_id="FUND_999", product_name="幻覺基金", reason="r")])

    tools = make_tools(paths, BadWhat())
    result = tools.run(TOOL_NBA, "C000104", SessionState())
    assert result.nba.status == "rejected"
    row = read_log(paths["tmp"] / "latency.jsonl")[0]
    assert row["what_attempts"] == 2 and row["what_llm_latency_sec"] is not None
