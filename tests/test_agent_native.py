"""Gemini native function-calling adapter: real google-genai types, fake connection."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from google.genai import types

from agent.agent_llm import (
    GeminiNativeAgentLLM,
    ToolCallRequest,
    ToolExecution,
    native_function_declarations,
)
from agent.agent_tools import TOOL_INSIGHT, TOOL_NAMES, TOOL_NBA, TOOL_TOP10


def model_turn(*parts) -> types.Content:
    return types.Content(role="model", parts=list(parts))


def call_part(name, args=None) -> types.Part:
    return types.Part(function_call=types.FunctionCall(name=name, args=args))


class FakeGemini:
    """Quacks like GeminiLLMClient (.client / .model) and replays scripted responses."""

    def __init__(self, *contents, error=None):
        self._contents = list(contents)
        self.requests: list[dict] = []
        self._error = error
        self.model = "fake-model"
        self.client = SimpleNamespace(models=SimpleNamespace(generate_content=self._generate))

    def _generate(self, model, contents, config):
        # Snapshot: the adapter keeps appending to the same list afterwards.
        self.requests.append({"model": model, "contents": list(contents), "config": config})
        if self._error:
            raise self._error
        content = self._contents.pop(0)
        candidates = [] if content is None else [SimpleNamespace(content=content)]
        return SimpleNamespace(candidates=candidates)


def test_declarations_cover_every_tool_and_require_customer_id_only_where_needed():
    declarations = {d.name: d for d in native_function_declarations()}
    assert set(declarations) == TOOL_NAMES
    assert declarations[TOOL_TOP10].parameters is None
    for name in (TOOL_INSIGHT, TOOL_NBA):
        assert declarations[name].parameters.required == ["customer_id"]
        assert "customer_id" in declarations[name].parameters.properties


def test_first_request_sends_the_question_the_tools_and_disables_automatic_calling():
    gemini = FakeGemini(model_turn(call_part(TOOL_TOP10)))
    llm = GeminiNativeAgentLLM(gemini)
    llm.start("CONTEXT: 今天要先聯繫誰？")
    llm.next([])

    request = gemini.requests[0]
    assert request["model"] == "fake-model"
    first = request["contents"][0]
    assert first.role == "user" and first.parts[0].text == "CONTEXT: 今天要先聯繫誰？"
    config = request["config"]
    assert {d.name for d in config.tools[0].function_declarations} == TOOL_NAMES
    assert config.automatic_function_calling.disable is True  # our code executes tools, not the SDK
    assert "工具路由器" in config.system_instruction


def test_function_call_becomes_a_tool_call_request():
    gemini = FakeGemini(model_turn(call_part(TOOL_NBA, {"customer_id": "C000104"})))
    llm = GeminiNativeAgentLLM(gemini)
    llm.start("q")
    decision = llm.next([])
    assert decision.tool_calls == [ToolCallRequest(tool_name=TOOL_NBA, customer_id="C000104")]
    assert decision.needs_more_steps is True and decision.reply == ""


def test_parallel_calls_and_missing_arguments_are_handled():
    gemini = FakeGemini(model_turn(call_part(TOOL_TOP10, None), call_part(TOOL_INSIGHT, {"customer_id": "C000518"})))
    llm = GeminiNativeAgentLLM(gemini)
    llm.start("q")
    decision = llm.next([])
    assert [(c.tool_name, c.customer_id) for c in decision.tool_calls] == [(TOOL_TOP10, ""), (TOOL_INSIGHT, "C000518")]


def test_text_only_turn_is_a_reply_with_no_tool_calls_and_thoughts_are_dropped():
    gemini = FakeGemini(model_turn(
        types.Part(text="(內部思考)", thought=True), types.Part(text="請提供客戶編號。"),
    ))
    llm = GeminiNativeAgentLLM(gemini)
    llm.start("q")
    decision = llm.next([])
    assert decision.tool_calls == [] and decision.reply == "請提供客戶編號。" and decision.needs_more_steps is False


def test_tool_results_go_back_as_function_responses_after_the_models_own_turn():
    first_turn = model_turn(call_part(TOOL_TOP10), call_part(TOOL_INSIGHT, {"customer_id": "C000104"}))
    gemini = FakeGemini(first_turn, model_turn(types.Part(text="完成")))
    llm = GeminiNativeAgentLLM(gemini)
    llm.start("q")
    llm.next([])
    llm.next([
        ToolExecution(call=ToolCallRequest(tool_name=TOOL_TOP10, customer_id=""), summary="SUMMARY-TOP"),
        ToolExecution(call=ToolCallRequest(tool_name=TOOL_INSIGHT, customer_id="C000104"), summary="SUMMARY-INS"),
    ])

    contents = gemini.requests[1]["contents"]
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert contents[1] is first_turn  # the model's turn is replayed unchanged (keeps thought signatures)
    responses = contents[2].parts
    assert [p.function_response.name for p in responses] == [TOOL_TOP10, TOOL_INSIGHT]  # same order as the calls
    assert [p.function_response.response for p in responses] == [{"result": "SUMMARY-TOP"}, {"result": "SUMMARY-INS"}]


def test_start_begins_a_fresh_conversation():
    gemini = FakeGemini(model_turn(types.Part(text="a")), model_turn(types.Part(text="b")))
    llm = GeminiNativeAgentLLM(gemini)
    llm.start("first")
    llm.next([])
    llm.start("second")
    llm.next([])
    assert len(gemini.requests[1]["contents"]) == 1
    assert gemini.requests[1]["contents"][0].parts[0].text == "second"


def test_response_without_candidates_raises_so_the_agent_can_report_it():
    llm = GeminiNativeAgentLLM(FakeGemini(None))
    llm.start("q")
    with pytest.raises(RuntimeError):
        llm.next([])


def test_transient_api_errors_are_retried(monkeypatch):
    class Transient(Exception):
        code = 503

    monkeypatch.setattr("insight.insight_generator.RETRY_DELAYS_SEC", (0.0, 0.0, 0.0))
    gemini = FakeGemini(error=Transient("503 UNAVAILABLE"))
    llm = GeminiNativeAgentLLM(gemini)
    llm.start("q")
    with pytest.raises(Transient):
        llm.next([])
    assert len(gemini.requests) == 4  # first try + 3 retries
