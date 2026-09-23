"""Headless run of the Streamlit app (no browser): fake LLM, real UI code.

Skipped where Streamlit is too old for streamlit.testing (or not installed).
"""

from __future__ import annotations

import pandas as pd
import pytest

try:
    from streamlit.testing.v1 import AppTest
except Exception as exc:  # not installed, too old, or a protobuf mismatch in a shared env
    pytest.skip(f"Streamlit's headless test tool is unavailable here: {type(exc).__name__}", allow_module_level=True)

from test_agent import make_row  # noqa: E402
from tools.customer_tool import EXPECTED_FIELDS  # noqa: E402


APP_SCRIPT = """
from pathlib import Path

from agent.agent_llm import AgentDecision, ToolCallRequest
from agent.agent_tools import AgentTools
from test_agent import FakeBackend, fake_retriever
from ui.app import SharedResources, run_app


def call(tool, cid=""):
    return AgentDecision(tool_calls=[ToolCallRequest(tool_name=tool, customer_id=cid)], reply="", needs_more_steps=False)


class RoutingBackend(FakeBackend):
    # Plays the LLM: routes by keywords in the question, otherwise asks for a customer id.

    def generate(self, prompt, response_model):
        if response_model is AgentDecision:
            question = prompt.split("【理專這次的問題】")[1].split("【本輪")[0]
            if "名單" in question or "聯繫誰" in question:
                return call("get_top_priority_customers")
            if "為什麼" in question:
                return call("get_customer_insight", "C000104")
            if "下一步" in question:
                return call("get_next_best_action", "C000104")
            return AgentDecision(tool_calls=[], reply="請提供客戶編號。", needs_more_steps=False)
        return super().generate(prompt, response_model)


backend = RoutingBackend(insight_bad=__INSIGHT_BAD__)
tmp = Path(__TMP__)
tools = AgentTools(
    backend, fake_retriever, source_path=Path(__ALL__), top10_path=Path(__TOP10__),
    insight_audit_log_path=tmp / "i.jsonl", nba_audit_log_path=tmp / "n.jsonl",
)
run_app(lambda: SharedResources(llm_client=backend, tools=tools))
"""

FAILING_SCRIPT = """
from ui.app import run_app


def broken():
    raise RuntimeError("No LLM API key set.")


run_app(broken)
"""


def write_script(tmp_path, source: str, name: str):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")  # UTF-8: the script contains Chinese
    return str(path)


@pytest.fixture
def files(tmp_path):
    rows = [
        make_row("C000104", 1, investment_intent_score=92.0),
        make_row("C000518", 2, investment_intent_score=92.0),
        make_row("C000301", 3, risk_level="Low"),
    ]
    df = pd.DataFrame(rows, columns=EXPECTED_FIELDS)
    all_path, top10_path = tmp_path / "all.csv", tmp_path / "top10.csv"
    df.to_csv(all_path, index=False)
    df.iloc[:2].to_csv(top10_path, index=False)
    return all_path, top10_path, tmp_path


def make_app(files, insight_bad=False) -> AppTest:
    all_path, top10_path, tmp = files
    source = (
        APP_SCRIPT.replace("__INSIGHT_BAD__", str(insight_bad))
        .replace("__TMP__", repr(str(tmp))).replace("__ALL__", repr(str(all_path))).replace("__TOP10__", repr(str(top10_path)))
    )
    return AppTest.from_file(write_script(tmp, source, "app_script.py"), default_timeout=30)


def texts(elements) -> str:
    return " ".join(str(e.value) for e in elements)


def test_first_screen_shows_title_priority_list_and_examples(files):
    at = make_app(files).run()
    assert not at.exception
    assert at.title[0].value == "Wealth RM Agent"
    assert [b.key for b in at.sidebar.button if str(b.key).startswith("pick_")] == ["pick_C000104", "pick_C000518"]
    assert "可以這樣問" in texts(at.markdown)


def test_asking_for_the_list_renders_a_table_inside_the_conversation(files):
    at = make_app(files).run()
    at.chat_input[0].set_value("今天要先聯繫誰？").run()
    assert not at.exception
    assert len(at.chat_message) == 2  # the question and the answer
    assert len(at.dataframe) >= 1
    assert "今日優先名單" in texts(at.markdown)


def test_clicking_a_customer_offers_quick_questions_and_answers_with_a_status_badge(files):
    at = make_app(files).run()
    at.sidebar.button(key="pick_C000104").click().run()
    assert "已選客戶" in texts(at.markdown)
    at.button(key="quick_why").click().run()
    assert not at.exception
    assert "C000104 為什麼值得聯繫？" in texts(at.chat_message[0].markdown)
    assert any("通過事實查核" in str(s.value) for s in at.success)
    assert "客戶投資意圖高" in texts(at.markdown)


def test_recommendation_shows_conditions_product_card_and_disclaimer(files):
    at = make_app(files).run()
    at.chat_input[0].set_value("C000104 下一步該怎麼做？").run()
    assert not at.exception
    shown = texts(at.markdown) + " " + texts(at.caption)
    assert "A2 基金產品介紹" in shown and "investment_intent_score=92.0 >= 70" in shown
    assert "Fund｜風險 High｜TWD" in shown
    assert "最終適合度" in shown  # disclaimer


def test_rejected_insight_is_shown_as_an_error_with_no_ai_text(files):
    at = make_app(files, insight_bad=True).run()
    at.chat_input[0].set_value("C000104 為什麼值得聯繫？").run()
    assert not at.exception
    assert any("未通過事實查核" in str(e.value) for e in at.error)
    assert "客戶投資意圖高" not in texts(at.markdown)


def test_unsupported_question_shows_the_clarification_and_no_disclaimer(files):
    at = make_app(files).run()
    at.chat_input[0].set_value("你好").run()
    assert not at.exception
    assert "請提供客戶編號。" in texts(at.markdown)
    assert "最終適合度" not in texts(at.caption)


def test_reset_button_clears_the_conversation(files):
    at = make_app(files).run()
    at.chat_input[0].set_value("今天要先聯繫誰？").run()
    assert len(at.chat_message) == 2
    at.sidebar.button[-1].click().run()  # "清除對話"
    assert len(at.chat_message) == 0


def test_startup_failure_shows_setup_help_instead_of_a_stack_trace(tmp_path):
    at = AppTest.from_file(write_script(tmp_path, FAILING_SCRIPT, "failing.py"), default_timeout=30).run()
    assert not at.exception
    assert any("無法啟動" in str(e.value) and "API key" in str(e.value) for e in at.error)
