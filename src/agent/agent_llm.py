"""Phase 4: the LLM side of the Agent -- how the model is asked to pick tools.

The Agent loop (rm_agent.py) only depends on `AgentLLMClient`:

    start(context)          begin one RM question
    next(executions)        give the model the previous round's tool results
                            (empty on the first call) and get its next decision

Two implementations behind that seam:

- GeminiNativeAgentLLM   provider-native function calling (Gemini): the tools
  are declared to the API, the model answers with function calls, and tool
  results are sent back as function responses in the same conversation.
- StructuredAgentLLM     works with any provider (Gemini or Anthropic): the
  model fills a JSON schema (`AgentDecision`) that names the tools to call.

Both give the model only a one-line SUMMARY of each tool result (ids, status),
never the insight / recommendation text, so the model has nothing to paraphrase
back at the RM -- the reply is rendered by code (render.py).

這個檔案的 LLM 只做一件事：決定「這次該呼叫哪個工具」，不負責回答理專的
問題本身。所以下面 AgentDecision 的 `reply` 欄位只在「不呼叫工具」時才會被
用來顯示（請理專補充資訊、或說明這項請求不支援），一旦有工具被呼叫，
這個 `reply` 就會被忽略，真正顯示給理專看的文字改由 render.py 組成。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from agent.agent_tools import TOOL_SPECS, ToolSpec  # noqa: E402
from insight.insight_generator import GeminiLLMClient, StructuredLLMClient, call_with_retry  # noqa: E402


class ToolCallRequest(BaseModel):
    tool_name: str
    customer_id: str  # empty string when the tool needs no customer id


class AgentDecision(BaseModel):
    """One round's decision. Also the JSON schema of the structured-output
    mode (no field defaults: Gemini's response schema does not support them)."""

    tool_calls: list[ToolCallRequest]
    reply: str  # only used when tool_calls is empty: a clarification or "unsupported" message
    needs_more_steps: bool  # true only if a later call depends on THESE calls' results


class ToolExecution(BaseModel):
    """What the model gets back for one of its calls: a one-line summary."""

    call: ToolCallRequest
    summary: str


class AgentLLMClient(Protocol):
    def start(self, context: str) -> None: ...

    def next(self, executions: list[ToolExecution]) -> AgentDecision: ...


# -- prompts -------------------------------------------------------------------

ROUTING_RULES = """- 你只負責「決定呼叫哪個工具」，不回答任何客戶或金融內容，也不評論工具結果。
- 客戶編號格式為 C 加 6 位數字（例如 C000104）。不可自行猜測或捏造客戶編號。
- 想知道今天先聯繫誰、優先名單 → get_top_priority_customers。
- 問客戶為什麼值得聯繫、客戶狀況 → get_customer_insight。
- 問下一步怎麼做、該建議什麼行動或產品 → get_next_best_action（它會自動先取得洞察，不必另外呼叫 get_customer_insight，除非使用者也想看洞察）。
- 使用者用「他」「這位客戶」指代時，使用「最近討論的客戶」；若無法確定是哪一位，不要呼叫工具，請使用者提供客戶編號。
- 使用者用「名單第一名」這類說法、但你還不知道客戶編號時：先只呼叫 get_top_priority_customers，看到結果後再用其中的客戶編號呼叫下一個工具。
- 請求與上述三項無關或不支援（例如下單、代為聯繫客戶、閒聊、沒有指定客戶的產品問答）：不要呼叫工具，用一到兩句話說明你能做什麼。
- 已經有足夠的工具結果可以回答使用者時：不要再呼叫工具。"""

STRUCTURED_OUTPUT_RULES = """- 要呼叫工具時填 tool_calls；不呼叫工具時 tool_calls 留空，並把要對使用者說的話（請使用者補充資訊、或說明不支援）寫在 reply。
- 只有當「之後的呼叫需要用到這一輪的結果」時 needs_more_steps 才設 true；否則一律 false。
- 已經有足夠的工具結果時：tool_calls 留空、reply 留空字串。"""

NATIVE_SYSTEM_PROMPT = f"""你是財富管理理專助理的「工具路由器」。根據理專的問題，決定要呼叫哪些工具。

規則（必須遵守）：
{ROUTING_RULES}
- 不呼叫工具時，直接用一到兩句話回覆使用者。已經取得足夠工具結果時，只回覆「完成」，不要複述或評論結果。"""


def build_context(user_message: str, discussed_line: str, recent_questions: list[str]) -> str:
    recent = "\n".join(f"- {line}" for line in recent_questions) or "（無）"
    return f"""【對話狀態】
{discussed_line}
最近的提問：
{recent}

【理專這次的問題】
{user_message}"""


def _tool_lines(specs: list[ToolSpec]) -> str:
    return "\n".join(
        f"- {spec.name}（{'需要 customer_id' if spec.needs_customer_id else '不需要 customer_id，customer_id 填空字串'}）：{spec.description}"
        for spec in specs
    )


def build_structured_prompt(context: str, summaries: list[str]) -> str:
    results = "\n".join(f"- {s}" for s in summaries) or "（本輪尚未呼叫任何工具）"
    return f"""你是財富管理理專助理的「工具路由器」。根據理專的問題，決定要呼叫哪些工具。

【可用工具】
{_tool_lines(TOOL_SPECS)}

{context}

【本輪已執行的工具結果摘要】
{results}

規則（必須遵守）：
{ROUTING_RULES}
{STRUCTURED_OUTPUT_RULES}
"""


# -- structured-output implementation (any provider) ------------------------------

class StructuredAgentLLM:
    """Adapts any StructuredLLMClient (Gemini / Anthropic clients) to AgentLLMClient."""

    def __init__(self, client: StructuredLLMClient) -> None:
        self._client = client
        self._context = ""
        self._summaries: list[str] = []

    def start(self, context: str) -> None:
        self._context = context
        self._summaries = []

    def next(self, executions: list[ToolExecution]) -> AgentDecision:
        self._summaries += [e.summary for e in executions]
        return self._client.generate(build_structured_prompt(self._context, self._summaries), AgentDecision)


# -- Gemini native function calling ----------------------------------------------

def native_function_declarations(specs: list[ToolSpec] | None = None) -> list:
    """The tool declarations sent to Gemini (name, description, parameters)."""
    from google.genai import types

    declarations = []
    for spec in specs if specs is not None else TOOL_SPECS:
        parameters = None
        if spec.needs_customer_id:
            parameters = types.Schema(
                type="OBJECT",
                properties={"customer_id": types.Schema(type="STRING", description="客戶編號，例如 C000104")},
                required=["customer_id"],
            )
        declarations.append(types.FunctionDeclaration(name=spec.name, description=spec.description, parameters=parameters))
    return declarations


class GeminiNativeAgentLLM:
    """Provider-native function calling with Gemini.

    Keeps the conversation (`contents`) across the rounds of ONE question. The
    model's own turn is appended exactly as returned -- Gemini 3 models attach
    thought signatures to function-call parts that must be sent back unchanged.
    Automatic function calling is disabled: our code, not the SDK, executes
    tools, so every guardrail in agent_tools.py / rm_agent.py still applies.
    """

    def __init__(self, gemini: GeminiLLMClient) -> None:
        self._client = gemini.client
        self._model = gemini.model
        self._contents: list = []

    def start(self, context: str) -> None:
        from google.genai import types

        self._contents = [types.Content(role="user", parts=[types.Part(text=context)])]

    def _config(self):
        from google.genai import types

        return types.GenerateContentConfig(
            system_instruction=NATIVE_SYSTEM_PROMPT,
            tools=[types.Tool(function_declarations=native_function_declarations())],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def next(self, executions: list[ToolExecution]) -> AgentDecision:
        from google.genai import types

        if executions:
            # One function response per function call the model made, in order.
            self._contents.append(types.Content(role="user", parts=[
                types.Part.from_function_response(name=e.call.tool_name, response={"result": e.summary})
                for e in executions
            ]))

        response = call_with_retry(
            lambda: self._client.models.generate_content(
                model=self._model, contents=self._contents, config=self._config()
            )
        )
        if not response.candidates or response.candidates[0].content is None:
            raise RuntimeError("Gemini returned no candidate content for the tool-calling request")
        content = response.candidates[0].content
        self._contents.append(content)

        parts = content.parts or []
        calls = [
            ToolCallRequest(tool_name=p.function_call.name, customer_id=str((p.function_call.args or {}).get("customer_id") or ""))
            for p in parts if getattr(p, "function_call", None)
        ]
        text = "".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False))
        return AgentDecision(tool_calls=calls, reply=text, needs_more_steps=bool(calls))
