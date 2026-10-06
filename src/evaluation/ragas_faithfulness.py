"""Phase 6: RAGAS Faithfulness scoring for WHY/WHAT free-text outputs.

Faithfulness measures whether a piece of generated free text is grounded in
a given context: it decomposes the text into atomic claims and checks, via
an LLM judge, whether each claim can be inferred from the context. This is
the semantic counterpart to the existing deterministic checks
(insight/fact_validator.py, rag/recommendation_validator.py), which only
cover structured fields -- see docs/evaluation_design.md for the full gap
analysis.

Upstream bug workaround: every currently-available ragas release
unconditionally imports `langchain_community.chat_models.vertexai.
ChatVertexAI` from `ragas/llms/base.py`, even though this module never uses
VertexAI. That import path was removed from langchain-community in its
0.4.2 release (ChatVertexAI moved to the separate langchain-google-vertexai
package), so `import ragas` crashes for every user on a current
langchain-community, regardless of ragas version or LLM provider (confirmed
upstream bug; a fix PR exists but is not released as of this writing).
Pinning an old langchain-community risks its own dependency conflicts with
the langchain-core version this project already uses elsewhere, so instead
this module stubs the dead import path out in `sys.modules` before `ragas`
is ever imported -- a self-contained, zero-new-dependency workaround. Once
ragas ships the real fix, `_stub_ragas_vertexai_import()` and its call
below can be deleted.
"""

from __future__ import annotations

import math
import os
import sys
import types
from pathlib import Path
from typing import Optional

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from insight.insight_generator import resolve_model  # noqa: E402

DEFAULT_JUDGE_MODEL = "gemini-3.5-flash"


def _stub_ragas_vertexai_import() -> None:
    """Register a fake langchain_community.chat_models.vertexai module so
    ragas/llms/base.py's `from ... import ChatVertexAI` succeeds without
    langchain-community actually providing it. Must run before the first
    `import ragas` / `from ragas... import ...` anywhere in the process."""
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return
    stub = types.ModuleType(module_name)
    stub.ChatVertexAI = type("ChatVertexAI", (), {})  # never instantiated; we use Gemini
    sys.modules[module_name] = stub


_stub_ragas_vertexai_import()


def resolve_judge_model(model: str | None = None) -> str:
    return resolve_model(model, "RAGAS_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)


def get_faithfulness_llm(model: str | None = None, api_key: str | None = None):
    """Build the ragas-wrapped Gemini LLM used as the Faithfulness judge.

    Model precedence: explicit argument > .env's RAGAS_JUDGE_MODEL >
    DEFAULT_JUDGE_MODEL -- same pattern as GeminiLLMClient / GEMINI_MODEL
    (insight_generator.py), so the judge model can be swapped independently
    of the generation model without a code change.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    from ragas.llms import LangchainLLMWrapper

    resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
    resolved_model = resolve_judge_model(model)
    chat_model = ChatGoogleGenerativeAI(model=resolved_model, google_api_key=resolved_key)
    return LangchainLLMWrapper(chat_model)


def score_faithfulness(response: str, contexts: list[str], llm, user_input: str) -> Optional[float]:
    """Faithfulness score in [0, 1]: fraction of atomic claims in `response`
    that can be inferred from `contexts`. 1.0 = fully grounded, lower =
    more unsupported claims.

    `user_input` is the question the response answers. ragas passes it to the
    judge when splitting the response into statements, so it must be a real
    question, not an empty string.

    Returns None when ragas finds no statements to judge (it reports NaN for
    that case); callers must treat None as "not scored", never as a number.

    Costs two real LLM calls via `llm` (statement split + verdicts) -- never
    call with a real judge from a test. Synchronous wrapper around ragas's
    async scoring method so callers stay synchronous.
    """
    import asyncio

    from ragas import SingleTurnSample
    from ragas.metrics import Faithfulness

    sample = SingleTurnSample(user_input=user_input, response=response, retrieved_contexts=contexts)
    metric = Faithfulness(llm=llm)
    return _nan_to_none(float(asyncio.run(metric.single_turn_ascore(sample))))


def _nan_to_none(score: float) -> Optional[float]:
    return None if math.isnan(score) else score


def main() -> None:
    """Minimal smoke test: one obviously-faithful and one obviously-unfaithful
    example, to confirm the ragas <-> LangChain <-> Gemini wiring actually
    works before building the full evaluation pipeline on top of it.

    Usage: python src/evaluation/ragas_faithfulness.py
    """
    from insight.insight_generator import load_env_file

    load_env_file()
    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY not set in .env")
        raise SystemExit(1)

    llm = get_faithfulness_llm()
    context = ["客戶總資產（AUM）為 800 萬元。", "客戶台幣活存占比為 72%。"]

    faithful_response = "客戶資產約 800 萬元，台幣活存占比約 72%。"
    unfaithful_response = "客戶資產約 800 萬元，且是金融業高階主管，非常適合高風險商品。"

    print("Scoring a faithful response (expect close to 1.0)...")
    question = "這位客戶的資產狀況如何？"
    score_ok = score_faithfulness(faithful_response, context, llm, question)
    print(f"  Faithfulness score: {score_ok}")

    print("Scoring an unfaithful response (expect well below 1.0)...")
    score_bad = score_faithfulness(unfaithful_response, context, llm, question)
    print(f"  Faithfulness score: {score_bad}")


if __name__ == "__main__":
    main()
