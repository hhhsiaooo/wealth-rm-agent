"""Phase 4: the Agent -- autonomous tool selection over the WHO / WHY / WHAT tools.

Loop for one RM question (max MAX_STEPS rounds):
    LLM decides -> which tool(s) to call, or a clarification / "unsupported" reply
    code runs the tools (agent/agent_tools.py) -> results
    the LLM gets a one-line summary of each result (never the content); it may
    decide again if a later call depends on them (e.g. it needed the Top-10 list
    to learn which customer "第一名" is)
    code renders the reply from the tool results (agent/render.py)

What the LLM may and may not do (docs/ai_product_design.md, section 八):
- may: choose tools, ask for a missing customer id, say a request is unsupported
- may not: write the answer. Once any tool ran, the LLM's free text is
  ignored and the reply is rendered by code from validated results.

How the LLM picks tools is behind `AgentLLMClient` (agent/agent_llm.py):
provider-native function calling for Gemini (default), or schema-constrained
structured output for any provider (Anthropic, or `--routing structured`).

Guardrails: step cap, tool-call cap, tool-name allowlist and customer_id
existence (agent_tools.py), duplicate calls skipped, LLM/API failure reported
not raised. A follow-up LLM round happens only after the Top-10 tool ran: it is
the only tool whose result supplies information (customer ids) for further
calls; insight / recommendation results end the question, which saves an LLM
call per question.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

from pydantic import BaseModel

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from agent.agent_llm import (  # noqa: E402,F401  (some names re-exported for callers/tests)
    AgentDecision,
    AgentLLMClient,
    GeminiNativeAgentLLM,
    StructuredAgentLLM,
    ToolCallRequest,
    ToolExecution,
    build_context,
    build_structured_prompt,
)
from agent.agent_tools import (  # noqa: E402
    TOOL_NAMES,
    TOOL_TOP10,
    AgentTools,
    LazyRetriever,
    SessionState,
    ToolResult,
    default_retriever_factory,
    normalize_customer_id,
)
from agent.render import render_response  # noqa: E402
from insight.insight_generator import GeminiLLMClient, load_env_file, select_llm_client  # noqa: E402
from rag.product_knowledge import ProductKnowledge  # noqa: E402

MAX_STEPS = 4
MAX_TOOL_CALLS = 5
HISTORY_TURNS = 4

DEFAULT_CLARIFICATION = (
    "我可以幫你：1) 列出今日優先聯繫名單；2) 說明某位客戶為什麼值得聯繫；3) 建議某位客戶的下一步行動。"
    "請告訴我想做哪一項，以及客戶編號（例如 C000104）。"
)
LLM_UNAVAILABLE = "目前無法連線到 LLM 判斷你的問題（{error}）。請稍後再試，或直接說出要查的客戶編號與項目。"


class AgentResponse(BaseModel):
    reply: str
    trace: list[str] = []
    tool_results: list[ToolResult] = []


def summarize_result(result: ToolResult) -> str:
    """One line for the LLM. Deliberately carries no insight/advice text --
    only what is needed to decide the next call (ids, statuses)."""
    target = f"({result.customer_id})" if result.customer_id else "()"
    if not result.ok:
        return f"{result.tool}{target} 失敗：{result.error}"
    if result.top10 is not None:
        ids = "、".join(f"第{r.priority_rank}名 {r.customer_id}" for r in result.top10)
        return f"{result.tool}{target} 完成，名單：{ids}"
    if result.insight is not None:
        return f"{result.tool}{target} 完成，洞察狀態：{result.insight.status}"
    if result.nba is not None:
        return f"{result.tool}{target} 完成，建議行動：{result.nba.selection.recommended_action}，狀態：{result.nba.status}"
    return f"{result.tool}{target} 完成"


class Agent:
    def __init__(
        self,
        llm: AgentLLMClient,
        tools: AgentTools,
        max_steps: int = MAX_STEPS,
        max_tool_calls: int = MAX_TOOL_CALLS,
        catalog: Optional[dict[str, ProductKnowledge]] = None,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._max_steps = max_steps
        self._max_tool_calls = max_tool_calls
        self._catalog = catalog
        self.session = SessionState()
        self._history: list[str] = []

    def reset(self) -> None:
        self.session = SessionState()
        self._history = []

    def _discussed_line(self) -> str:
        if not self.session.discussed:
            return "本次對話尚未討論任何客戶。"
        discussed = "、".join(
            f"{cid}（{'已有洞察' if cid in self.session.insights and self.session.insights[cid].insight else '尚無洞察'}）"
            for cid in self.session.discussed
        )
        return f"本次對話已討論的客戶（最近討論的在最後）：{discussed}"

    def ask(self, user_message: str, on_event: Optional[Callable[[str], None]] = None) -> AgentResponse:
        """一個問題的完整流程：問 LLM 要呼叫哪個工具 → 執行工具 → 視需要再問一輪
        → 最後把工具結果交給 render_response() 組成回覆。LLM 全程只負責「選工具」，
        從來不會自己寫出理專看到的最終文字。"""
        trace: list[str] = []

        def note(text: str) -> None:
            trace.append(text)
            if on_event:
                on_event(text)

        self._llm.start(build_context(user_message, self._discussed_line(), self._history[-HISTORY_TURNS:]))

        results: list[ToolResult] = []
        seen: set[tuple[str, str]] = set()
        executions: list[ToolExecution] = []  # previous round's results, handed to the LLM next round
        stopped_by_cap = False

        for step in range(1, self._max_steps + 1):
            try:
                decision = self._llm.next(executions)
            except Exception as exc:  # noqa: BLE001 - an LLM/API failure must not crash the conversation
                note(f"LLM 判斷失敗：{type(exc).__name__}")
                if results:
                    break  # still show what already ran
                return self._finish(user_message, LLM_UNAVAILABLE.format(error=f"{type(exc).__name__}: {exc}"), trace, results)

            if not decision.tool_calls:
                if not results:  # nothing ran: the LLM's message is a clarification / "unsupported" reply
                    return self._finish(user_message, decision.reply.strip() or DEFAULT_CLARIFICATION, trace, results)
                break  # tools ran: the LLM's text is ignored, the reply is rendered from results

            executions = []
            executed = 0
            fed_next_call = False  # did a tool run whose result can inform another call?
            for call in decision.tool_calls:
                # Every call the model made gets an entry (native function calling
                # requires one response per call), even if we skip it.
                if len(results) >= self._max_tool_calls:
                    if not stopped_by_cap:
                        note(f"已達單次提問的工具呼叫上限（{self._max_tool_calls}），停止")
                    stopped_by_cap = True
                    executions.append(ToolExecution(call=call, summary="未執行：已達單次提問的工具呼叫上限"))
                    continue
                key = (call.tool_name, normalize_customer_id(call.customer_id))
                if key in seen:
                    note(f"略過重複呼叫：{call.tool_name} {key[1]}".rstrip())
                    executions.append(ToolExecution(call=call, summary="已略過：與先前的呼叫重複"))
                    continue
                seen.add(key)
                label = f"{call.tool_name}({key[1]})" if key[1] else f"{call.tool_name}()"
                if call.tool_name not in TOOL_NAMES:
                    note(f"拒絕不在白名單的工具：{label}")
                else:
                    note(f"[第 {step} 輪] 呼叫工具：{label}")
                result = self._tools.run(call.tool_name, call.customer_id, self.session)
                note(f"  → {'成功' if result.ok else '失敗：' + str(result.error)}")
                results.append(result)
                executions.append(ToolExecution(call=call, summary=summarize_result(result)))
                executed += 1
                fed_next_call = fed_next_call or (result.ok and result.tool == TOOL_TOP10)

            if stopped_by_cap or executed == 0 or not (decision.needs_more_steps and fed_next_call):
                break
        else:
            note(f"已達最多 {self._max_steps} 輪，停止")

        if not results:
            return self._finish(user_message, DEFAULT_CLARIFICATION, trace, results)
        # 關鍵一步：理專最後看到的文字是 render_response()（render.py）依 results
        # 組出來的，不是 LLM 這一輪講的話——LLM 選完工具之後就沒有機會再插手文字內容。
        return self._finish(user_message, render_response(results, self._catalog), trace, results)

    def _finish(self, user_message: str, reply: str, trace: list[str], results: list[ToolResult]) -> AgentResponse:
        used = "、".join(r.tool for r in results) or "未呼叫工具"
        self._history.append(f"{user_message}（{used}）")
        return AgentResponse(reply=reply, trace=trace, tool_results=results)


# -- CLI -----------------------------------------------------------------------

HELP_TEXT = """可以這樣問：
  今天要先聯繫誰？
  C000104 為什麼值得聯繫？
  C000104 下一步該怎麼做？          （會自動先取得洞察）
  名單第一名的客戶，下一步怎麼做？   （Agent 會先看名單再查該客戶）
指令：/reset 清除對話狀態　/help 說明　exit 離開"""
EXIT_WORDS = {"exit", "quit", "q", "離開", "結束"}


def build_llm_adapter(llm_client, routing: str = "native") -> tuple[AgentLLMClient, str]:
    """Pick how the LLM chooses tools. Returns (adapter, human-readable description)."""
    if routing == "native" and isinstance(llm_client, GeminiLLMClient):
        return GeminiNativeAgentLLM(llm_client), f"Gemini 原生 function calling（{llm_client.model}）"
    note = "" if routing == "structured" else "（此供應商沒有原生轉接器，改用結構化輸出）"
    return StructuredAgentLLM(llm_client), f"結構化輸出工具路由{note}"


def build_default_agent(routing: str = "native") -> tuple[Agent, str]:
    """Returns (agent, description of the routing mode actually used)."""
    load_env_file()
    llm_client = select_llm_client()
    tools = AgentTools(llm_client, LazyRetriever(default_retriever_factory))
    adapter, description = build_llm_adapter(llm_client, routing)
    return Agent(adapter, tools), description


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Chat with the agent in the terminal.

        python src/agent/rm_agent.py                       # interactive
        python src/agent/rm_agent.py --ask "今天要先聯繫誰？"   # one question, then exit
        python src/agent/rm_agent.py --quiet               # hide the [agent] tool-call trace
        python src/agent/rm_agent.py --routing structured  # structured-output routing instead of native function calling

    Needs GEMINI_API_KEY or ANTHROPIC_API_KEY in .env (model names: GEMINI_MODEL /
    ANTHROPIC_MODEL). Product retrieval (Chroma + Gemini embeddings) loads only
    when a recommendation is asked for; run in an environment where google-genai
    and chromadb can share a process."""
    parser = argparse.ArgumentParser(description="Wealth RM Agent (WHO / WHY / WHAT)")
    parser.add_argument("--ask", help="ask one question and exit")
    parser.add_argument("--quiet", action="store_true", help="do not print the tool-call trace")
    parser.add_argument("--routing", choices=["native", "structured"], default="native",
                        help="how the LLM picks tools (default: native function calling on Gemini)")
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to a legacy codepage
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    agent, routing_description = build_default_agent(args.routing)
    on_event = None if args.quiet else (lambda text: print(f"  [agent] {text}", flush=True))

    def answer(question: str) -> None:
        response = agent.ask(question, on_event=on_event)
        print(f"\n{response.reply}\n", flush=True)

    if args.ask:
        answer(args.ask)
        return

    print(f"Wealth RM Agent — 工具選擇方式：{routing_description}")
    print("輸入問題開始，/help 看範例，exit 離開。")
    while True:
        try:
            question = input("理專> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in EXIT_WORDS:
            break
        if question == "/help":
            print(HELP_TEXT)
        elif question == "/reset":
            agent.reset()
            print("已清除對話狀態。")
        else:
            answer(question)


if __name__ == "__main__":
    main()
