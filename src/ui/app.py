"""Phase 5 (UI): Streamlit chat app for the RM assistant.

    streamlit run src/ui/app.py

Layout: sidebar = today's priority list (click a customer to get quick
questions) + reset; main = chat. Each answer is drawn from validated tool
results (ui/view_models.py): tables for the list, status badges for the
fact-check outcome, expandable evidence fields, and an expandable "what the
Agent did" trace. The disclaimer stays visible.

Streamlit re-runs this whole script on every interaction, so:
- the Agent (conversation memory) lives in st.session_state, one per browser tab;
- heavy shared objects (LLM client, lazy retriever) are built once with
  st.cache_resource and shared across tabs. The Agent's LLM adapter keeps
  per-question conversation state, so it is created per session, not shared.

簡單說：這個檔案只負責「畫面」，本身不呼叫 LLM、不判斷行動、不做任何業務
邏輯——理專打字問問題後，實際工作都丟給 Agent（agent/rm_agent.py）做，這裡
只是把 Agent 回傳的結果（經過 ui/view_models.py 轉換）畫成 Streamlit 元件。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

import streamlit as st
from pydantic import BaseModel

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from agent.agent_tools import TOOL_TOP10, AgentTools, LazyRetriever, SessionState, default_retriever_factory  # noqa: E402
from agent.rm_agent import Agent, build_llm_adapter  # noqa: E402
from insight.insight_generator import load_env_file, select_llm_client  # noqa: E402
from ui.view_models import (  # noqa: E402
    DISCLAIMER,
    TOP_LIST_COLUMNS,
    Block,
    ErrorBlock,
    InsightBlock,
    RecommendationBlock,
    StatusBadge,
    TextBlock,
    TopListBlock,
    blocks_from_response,
    shows_disclaimer,
    top_list_rows,
)

PAGE_TITLE = "Wealth RM Agent"
EXAMPLE_QUESTIONS = [
    "今天要先聯繫誰？",
    "名單第一名的客戶，下一步怎麼做？",
    "C000104 為什麼值得聯繫？",
]


class SharedResources(BaseModel):
    """Built once per server process and shared by every browser tab."""

    model_config = {"arbitrary_types_allowed": True}

    llm_client: object
    tools: AgentTools


def build_shared_resources() -> SharedResources:
    load_env_file()
    llm_client = select_llm_client()  # raises with setup instructions when no API key is configured
    return SharedResources(llm_client=llm_client, tools=AgentTools(llm_client, LazyRetriever(default_retriever_factory)))


@st.cache_resource(show_spinner="準備 LLM 連線…")
def get_shared_resources() -> SharedResources:
    return build_shared_resources()


# -- rendering blocks --------------------------------------------------------------

def render_badge(badge: StatusBadge) -> None:
    {"success": st.success, "warning": st.warning, "error": st.error}[badge.tone](badge.label)


def render_block(block: Block) -> None:
    if isinstance(block, TopListBlock):
        st.markdown(f"#### 今日優先名單（共 {len(block.rows)} 位，同分並列者皆列出）")
        st.dataframe(block.rows, hide_index=True, column_order=TOP_LIST_COLUMNS)
    elif isinstance(block, InsightBlock):
        title = f"#### 客戶洞察　{block.customer_id}"
        if block.priority_rank is not None:
            title += f"　（優先排名 {block.priority_rank}，優先分數 {block.priority_score:g}）"
        st.markdown(title)
        render_badge(block.status)
        if block.text:
            st.write(block.text)
        if block.message:
            st.write(block.message)
        if block.key_fields:
            with st.expander("依據欄位（皆已與客戶原始資料比對）"):
                st.dataframe([f.model_dump() for f in block.key_fields], hide_index=True)
    elif isinstance(block, RecommendationBlock):
        st.markdown(f"#### 行動建議　{block.customer_id}")
        st.markdown(f"**建議行動：{block.action_id} {block.action_name}**")
        with st.expander("規則判斷：成立條件", expanded=True):
            for condition in block.conditions or ["（無）"]:
                st.markdown(f"- {condition}")
        render_badge(block.status)
        if block.explanation:
            st.write(block.explanation)
        if block.message:
            st.write(block.message)
        if block.products:
            st.markdown("**候選產品**（來自產品知識庫，仍需進一步確認適合度）")
            for product in block.products:
                with st.container(border=True):
                    st.markdown(f"**{product.name}**　`{product.product_id}`")
                    st.caption(product.meta)
                    st.write(product.reason)
        if block.insight_note:
            st.caption(block.insight_note)
    elif isinstance(block, ErrorBlock):
        st.error(block.message)
    elif isinstance(block, TextBlock):
        st.write(block.text)


def render_assistant_message(blocks: list[Block], trace: Optional[list[str]] = None) -> None:
    for block in blocks:
        render_block(block)
    if shows_disclaimer(blocks):
        st.caption(DISCLAIMER)
    if trace:
        with st.expander("Agent 做了什麼（工具呼叫過程）"):
            for line in trace:
                st.text(line)


# -- app -------------------------------------------------------------------------

def _init_state(shared: SharedResources) -> Agent:
    if "agent" not in st.session_state:
        adapter, description = build_llm_adapter(shared.llm_client)
        st.session_state.agent = Agent(adapter, shared.tools)
        st.session_state.routing_description = description
        st.session_state.messages = []
        st.session_state.selected_customer = None
    return st.session_state.agent


def run_app(get_shared: Callable[[], SharedResources] = get_shared_resources) -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon="💼", layout="wide")

    try:
        shared = get_shared()
    except Exception as exc:  # noqa: BLE001 - most likely a missing API key: show setup help, not a stack trace
        st.title(PAGE_TITLE)
        st.error(f"無法啟動：{exc}")
        st.stop()
        return

    agent = _init_state(shared)
    asked: Optional[str] = None

    # Sidebar: WHO -- computed by code, no LLM, so it costs nothing to show.
    with st.sidebar:
        st.header("今日優先名單")
        top = shared.tools.run(TOOL_TOP10, None, SessionState())
        if top.ok:
            for row in top_list_rows(top):
                label = f"{row['排名']:>2}　{row['客戶編號']}　{row['優先分數']:.1f}"
                if st.button(label, key=f"pick_{row['客戶編號']}"):
                    st.session_state.selected_customer = row["客戶編號"]
        else:
            st.error(top.error)
        st.caption("點一位客戶，可直接帶入常用問題。")
        st.divider()
        if st.button("清除對話"):
            agent.reset()
            st.session_state.messages = []
            st.session_state.selected_customer = None
            st.rerun()
        st.caption(f"工具選擇方式：{st.session_state.routing_description}")

    st.title(PAGE_TITLE)
    st.caption("財富管理理專決策支援：WHO 該聯繫誰 → WHY 為什麼 → WHAT 下一步做什麼。所有資料為虛構的 Synthetic Data。")

    selected = st.session_state.selected_customer
    if selected:
        with st.container(border=True):
            st.markdown(f"已選客戶：**{selected}**")
            col_why, col_what = st.columns(2)
            if col_why.button("為什麼值得聯繫？", key="quick_why"):
                asked = f"{selected} 為什麼值得聯繫？"
            if col_what.button("下一步該怎麼做？", key="quick_what"):
                asked = f"{selected} 下一步該怎麼做？"
    elif not st.session_state.messages:
        st.markdown("**可以這樣問：**")
        for example in EXAMPLE_QUESTIONS:
            st.markdown(f"- {example}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["text"])
            else:
                render_assistant_message(message["blocks"], message["trace"])

    typed = st.chat_input("輸入問題，例如：C000104 下一步該怎麼做？")
    question = asked or typed
    if not question:
        return

    st.session_state.messages.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.status("Agent 處理中…", expanded=True) as status:
            try:
                # 真正的工作在這一行發生：Agent 決定要呼叫哪些工具（WHO／WHY／
                # WHAT）、執行它們，回傳已經查核過的結果——UI 本身不參與判斷。
                response = agent.ask(question, on_event=lambda line: st.text(line))
                blocks = blocks_from_response(response)
                trace = response.trace
                status.update(label="Agent 已完成（展開可看過程）", state="complete", expanded=False)
            except Exception as exc:  # noqa: BLE001 - the Agent should not raise; if it does, show it, do not crash the page
                blocks, trace = [ErrorBlock(message=f"發生未預期的錯誤：{type(exc).__name__}: {exc}")], []
                status.update(label="發生錯誤", state="error", expanded=True)
        render_assistant_message(blocks, trace=None)  # the collapsed status above already holds the trace
    st.session_state.messages.append({"role": "assistant", "blocks": blocks, "trace": trace})


if __name__ == "__main__":
    run_app()
