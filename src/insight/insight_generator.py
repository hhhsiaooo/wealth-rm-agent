"""Phase 2 (WHY), Step 5: LLM-based Customer Insight generation.

Flow:
    get_customer_profile(customer_id)
        -> complete 43-field Customer Record
        -> LLM
        -> Relevant Fields + Customer Insight (CustomerInsight)

This module builds the prompt, calls an injected LLM client, and validates
the response's STRUCTURE against the CustomerInsight schema. It does NOT
cross-check the LLM's selected field values against the source record --
that Fact Validation step is implemented separately (fact_validator.py, not
this step). It also does not modify WHO ranking logic and does not
implement WHAT / RAG / Agent orchestration.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from tools.customer_tool import CustomerDataError, TOP10_PATH, get_customer_profile  # noqa: E402

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class KeyField(BaseModel):
    field: str
    value: Any


class CustomerInsight(BaseModel):
    customer_id: str
    priority_rank: int
    priority_score: float
    key_fields: list[KeyField] = Field(min_length=1)
    insight: str


class InsightGenerationError(Exception):
    """Raised when the LLM response cannot be turned into a valid CustomerInsight."""


class LLMClient(Protocol):
    """Anything that turns a prompt into a validated CustomerInsight.

    Swapping LLM providers means writing a new class with this __call__
    signature -- generate_customer_insight() does not depend on any
    specific vendor SDK.
    """

    def __call__(self, prompt: str) -> CustomerInsight: ...


def load_env_file(path: Path = ENV_FILE) -> None:
    """Minimal .env loader (stdlib only): sets os.environ for KEY=VALUE
    lines that aren't already set in the environment. Existing env vars
    always win, so a real deployment env is never overridden by .env."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


BUSINESS_QUESTION = "為什麼這位客戶值得優先聯繫？"

MUST_NOT_RULES = """- 不可捏造資料中不存在的事實。
- 不可修改資料中的數值。
- 不可假設資料中沒有出現的欄位值。
- 不可創造新的業務規則。
- 不可重新計算 Priority Score，customer_id / priority_rank / priority_score
  必須直接複製上述資料中的原始值，不得更動。
- 不可推薦任何金融商品。
- 不可做出最終適合度判斷。
- key_fields 中每個 "field" 必須完全等於上述資料的欄位名稱，"value" 必須
  完全等於上述資料中的原始值，不可修改、不可四捨五入、不可換算單位。"""


def _format_profile_as_json_lines(profile: dict[str, Any]) -> str:
    return "\n".join(
        f'  "{k}": {v!r}' if isinstance(v, str) else f'  "{k}": {v}'
        for k, v in profile.items()
    )


def build_prompt(profile: dict[str, Any]) -> str:
    fields_json = _format_profile_as_json_lines(profile)

    return f"""你是財富管理理專的客戶洞察助理。以下是一位客戶的完整結構化資料（43個欄位），
請根據這份資料回答：「{BUSINESS_QUESTION}」

客戶完整資料（來源：WHO Priority Ranking 輸出）：
{{
{fields_json}
}}

請你：
1. 從上述資料中找出與「{BUSINESS_QUESTION}」相關的欄位。
2. 找出對這位客戶 Priority Score 貢獻最大的關鍵訊號。
3. 將相關的客戶事實組織成一段簡潔的 Customer Insight（繁體中文）。
4. 說明這些事實為什麼讓這位客戶值得理專關注。
5. 回傳你用來支持 insight 的欄位清單（key_fields）。

規則（必須遵守）：
{MUST_NOT_RULES}
"""


def build_correction_prompt(
    profile: dict[str, Any],
    previous_insight: CustomerInsight,
    issue_details: list[str],
) -> str:
    """Feed the specific grounding failures back to the LLM and ask it to
    correct them -- a corrective retry, not a blind one. Used by
    fact_validator.generate_grounded_customer_insight() when the first
    attempt fails fact validation."""
    fields_json = _format_profile_as_json_lines(profile)
    issues_text = "\n".join(f"- {issue}" for issue in issue_details)

    return f"""你先前針對這位客戶產生了以下 Customer Insight，但事實查核（Fact Validation）發現有問題：

先前的回答：
{previous_insight.model_dump_json()}

發現的問題：
{issues_text}

請根據以下客戶完整資料重新回答：「{BUSINESS_QUESTION}」，並修正上述所有問題。

客戶完整資料（來源：WHO Priority Ranking 輸出）：
{{
{fields_json}
}}

規則（必須遵守，與先前相同）：
{MUST_NOT_RULES}
"""


class AnthropicLLMClient:
    """Real LLM client backed by the Anthropic API.

    Uses structured outputs (messages.parse(output_format=CustomerInsight))
    so the API guarantees a schema-valid response -- no manual JSON
    parsing/validation is done in this module.
    """

    def __init__(self, model: str = DEFAULT_ANTHROPIC_MODEL, api_key: str | None = None) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def __call__(self, prompt: str) -> CustomerInsight:
        response = self._client.messages.parse(
            model=self._model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
            output_format=CustomerInsight,
        )
        if response.parsed_output is None:
            raise InsightGenerationError(
                f"Anthropic API did not return a parsed CustomerInsight "
                f"(stop_reason={response.stop_reason!r})"
            )
        return response.parsed_output


class GeminiLLMClient:
    """Free-tier-eligible LLM client backed by the Google Gemini API.

    Uses structured output (response_schema=CustomerInsight) -- the SDK
    validates the model's JSON response against the schema and exposes it
    as a parsed Pydantic instance via response.parsed.
    """

    def __init__(self, model: str = DEFAULT_GEMINI_MODEL, api_key: str | None = None) -> None:
        from google import genai

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._client = genai.Client(api_key=resolved_key)
        self._model = model

    def __call__(self, prompt: str) -> CustomerInsight:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CustomerInsight,
            ),
        )
        if response.parsed is None:
            raise InsightGenerationError(
                "Gemini API did not return a parsed CustomerInsight "
                f"(text={response.text!r})"
            )
        return response.parsed


def select_llm_client() -> LLMClient:
    """Pick an LLM client based on which API key is configured, preferring
    the free Gemini tier when both are available. Shared by this module's
    main() and fact_validator.py's main() so the provider-selection logic
    lives in one place."""
    if os.environ.get("GEMINI_API_KEY"):
        return GeminiLLMClient()
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicLLMClient()
    raise InsightGenerationError(
        "No LLM API key set. Add ONE of the following to "
        f"{ENV_FILE} (see .env.example):\n"
        "  GEMINI_API_KEY     - free tier, get one at https://aistudio.google.com/apikey\n"
        "  ANTHROPIC_API_KEY  - paid, get one at https://console.anthropic.com/settings/keys"
    )


def generate_customer_insight(
    customer_id: str,
    llm_client: LLMClient,
    source_path: Path = TOP10_PATH,
) -> CustomerInsight:
    """Retrieve one customer's 43-field record and generate a CustomerInsight.

    Raises:
        CustomerNotFoundError / CustomerSchemaError / CustomerDataError:
            propagated unchanged from get_customer_profile().
        InsightGenerationError: if llm_client did not return a valid
            CustomerInsight instance.
    """
    profile = get_customer_profile(customer_id, source_path=source_path)
    prompt = build_prompt(profile)

    insight = llm_client(prompt)

    if not isinstance(insight, CustomerInsight):
        raise InsightGenerationError(
            f"llm_client returned {type(insight).__name__}, expected CustomerInsight"
        )

    return insight


def main() -> None:
    load_env_file()

    if len(sys.argv) > 1:
        customer_id = sys.argv[1]
    else:
        import pandas as pd

        df = pd.read_csv(TOP10_PATH)
        customer_id = df.loc[df["priority_rank"].idxmin(), "customer_id"]

    try:
        llm_client = select_llm_client()
        insight = generate_customer_insight(customer_id, llm_client)
    except CustomerDataError as e:
        print(f"Could not load customer data: {e}")
        raise SystemExit(1)
    except InsightGenerationError as e:
        print(f"{e}")
        raise SystemExit(1)

    print(insight.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
