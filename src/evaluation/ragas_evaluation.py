"""Phase 6: RAGAS Faithfulness evaluation for WHY/WHAT free-text outputs.

Scores the one gap the existing WHY/WHAT evaluation never covers (see
docs/evaluation_design.md): whether the LLM's free-text prose is actually
grounded in the material it was given, as opposed to the structured-field
checks in insight/fact_validator.py and rag/recommendation_validator.py.

Each text is paired with its OWN context, not one shared blob:
- WHY `insight`            vs. the customer's full 43-field record
- WHAT `explanation`       vs. the recommended action's matched conditions
                                + the WHY insight (same material the prompt
                                actually showed the LLM, see
                                rag/nba_generator.build_nba_prompt)
- WHAT each `reason`       vs. ONLY that product's own catalog data -- not
                                the other candidate products, and not the
                                customer record

Scores the RAW (first-attempt) generation only, consistent with how
why_evaluation.py / what_evaluation.py's "raw" metrics measure LLM/prompt
quality on the first attempt -- this also avoids doubling an already
expensive pipeline with the grounding policy's corrective retry, which
Faithfulness doesn't need.

Cost note: each customer costs 2 generation calls (WHY + WHAT) + 1 embedding
call (retrieval) + 2-5 judge calls (insight, explanation, one per product).
Keep --per-action small; this is far more expensive per customer than
why_evaluation.py / what_evaluation.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd
from pydantic import BaseModel

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from evaluation.ragas_faithfulness import get_faithfulness_llm, resolve_judge_model, score_faithfulness  # noqa: E402
from evaluation.what_evaluation import select_evaluation_customers  # noqa: E402
from insight.insight_generator import (  # noqa: E402
    BUSINESS_QUESTION,
    GeminiLLMClient,
    LLMClient,
    StructuredLLMClient,
    generate_customer_insight,
    load_env_file,
    select_llm_client,
)
from insight.insight_generator import _format_profile_as_json_lines  # noqa: E402
from rag.nba_generator import (  # noqa: E402
    _format_product,
    _generate as _generate_nba_llm_output,
    assemble_next_best_action,
    build_nba_prompt,
    prepare_nba_context,
)
from rag.product_knowledge import ProductKnowledge  # noqa: E402
from rag.recommendation_validator import default_catalog  # noqa: E402
from rag.retrieval import ProductRetriever  # noqa: E402
from tools.customer_tool import ALL_PATH, get_customer_profile  # noqa: E402

OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"
REPORT_PATH = OUTPUTS_DIR / "ragas_evaluation_report.csv"
SUMMARY_PATH = OUTPUTS_DIR / "ragas_evaluation_summary.json"
CHECKPOINT_PATH = OUTPUTS_DIR / "ragas_evaluation_checkpoint.jsonl"
REJUDGE_REPORT_PATH = OUTPUTS_DIR / "ragas_rejudge_report.csv"
REJUDGE_SUMMARY_PATH = OUTPUTS_DIR / "ragas_rejudge_summary.json"

# far more expensive per customer than what_evaluation.py; default small
PER_ACTION_DEFAULT = 1


# -- schemas -------------------------------------------------------------------

class FaithfulnessRecord(BaseModel):
    stage: Literal["WHY", "WHAT"]
    text_type: Literal["insight", "explanation", "reason"]
    customer_id: str
    recommended_action: Optional[str] = None  # WHAT only
    product_id: Optional[str] = None          # reason only
    # None = judge output unparseable after one retry
    faithfulness: Optional[float]
    # the judged generation, kept so the judge can be re-run alone
    response_text: Optional[str] = None
    # JSON list[str] of the contexts it was judged against
    contexts_json: Optional[str] = None


class RagasFaithfulnessSummary(BaseModel):
    total_customers: int
    total_scored_texts: int
    judge_failures: int
    mean_faithfulness_overall: Optional[float]
    mean_faithfulness_why_insight: Optional[float]
    mean_faithfulness_what_explanation: Optional[float]
    mean_faithfulness_what_reason: Optional[float]
    # set by main(); baseline comparisons need this
    generation_model: Optional[str] = None
    judge_model: Optional[str] = None


# -- pure aggregation (unit-tested with synthetic records) ----------------------

def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def compute_summary(records: list[FaithfulnessRecord]) -> RagasFaithfulnessSummary:
    if not records:
        raise ValueError("cannot summarize an empty list of records")

    scored = [r for r in records if r.faithfulness is not None]
    by_type: dict[str, list[float]] = {
        "insight": [], "explanation": [], "reason": []}
    for r in scored:
        by_type[r.text_type].append(r.faithfulness)

    return RagasFaithfulnessSummary(
        total_customers=len({r.customer_id for r in records}),
        total_scored_texts=len(scored),
        judge_failures=len(records) - len(scored),
        mean_faithfulness_overall=_mean([r.faithfulness for r in scored]),
        mean_faithfulness_why_insight=_mean(by_type["insight"]),
        mean_faithfulness_what_explanation=_mean(by_type["explanation"]),
        mean_faithfulness_what_reason=_mean(by_type["reason"]),
    )


# -- context assembly (pure given its inputs; kept separate so tests don't need
#    a real LLM to check WHICH context gets paired with which text) -------------

def why_context(profile: dict[str, Any]) -> list[str]:
    """Same rendering insight_generator.build_prompt() used to show the LLM
    the customer record, so the judge sees equivalent information."""
    return [_format_profile_as_json_lines(profile)]


def what_explanation_context(matched_conditions: list[str], insight_text: Optional[str]) -> list[str]:
    contexts = list(matched_conditions)
    if insight_text:
        contexts.append(insight_text)
    return contexts


def what_reason_context(product: ProductKnowledge, customer_risk_level: str) -> list[str]:
    """This product's own data, plus the one customer attribute the reason is
    allowed to compare against (risk level, which the reason prompt asks the
    LLM to state). Never the other candidates, never the rest of the customer
    record: other products or unrelated customer fields could 'support' a
    claim they shouldn't."""
    return [f"客戶風險屬性：{customer_risk_level}", _format_product(product)]


# -- real generation + scoring (impure: real LLM/embedding/judge calls) ---------

def _generate_raw_nba(
    customer_id: str,
    llm_client: StructuredLLMClient,
    retriever: ProductRetriever,
    insight_text: Optional[str],
    source_path: Path,
    catalog: dict[str, ProductKnowledge],
):
    """Rule layer -> retrieval -> LLM, WITHOUT the grounding policy's
    validate/retry loop (nba_generator.generate_grounded_next_best_action).
    Faithfulness scores raw generation quality; it doesn't need the retry."""
    ctx = prepare_nba_context(customer_id, retriever,
                              insight_text, source_path)
    output = _generate_nba_llm_output(
        llm_client, build_nba_prompt(ctx, catalog))
    nba = assemble_next_best_action(ctx, output)
    return ctx, nba


WHAT_EXPLANATION_QUESTION = "為什麼建議此行動？"
WHAT_REASON_QUESTION = "為什麼此商品可作為候選？"


def question_for(text_type: str) -> str:
    """The question each free-text field answers. ragas uses it when splitting
    the text into statements, so it must match what the text was written for."""
    return {
        "insight": BUSINESS_QUESTION,
        "explanation": WHAT_EXPLANATION_QUESTION,
        "reason": WHAT_REASON_QUESTION,
    }[text_type]


def _score_with_retry(response: str, contexts: list[str], judge_llm, question: str) -> Optional[float]:
    """Judge output that isn't valid JSON (seen with gemini-3.5-flash-lite: bad
    \\u escapes inside Chinese text) is retried once, then recorded as None
    instead of aborting the whole run. Anything else -- quota errors, network
    errors -- propagates, so a daily-quota stop still stops the run. A None
    score (ragas found no statements) is returned as-is and counted as a failure."""
    from langchain_core.exceptions import OutputParserException

    for attempt in range(2):
        try:
            return score_faithfulness(response, contexts, judge_llm, question)
        except OutputParserException:
            print(
                f"  judge output unparseable (attempt {attempt + 1}/2)", flush=True)
    return None


def _dump_contexts(contexts: list[str]) -> str:
    return json.dumps(contexts, ensure_ascii=False)


def score_customer_faithfulness(
    customer_id: str,
    llm_client: LLMClient,
    judge_llm,
    retriever: ProductRetriever,
    source_path: Path = ALL_PATH,
    catalog: Optional[dict[str, ProductKnowledge]] = None,
    delay_between_calls_sec: float = 0.0,
) -> list[FaithfulnessRecord]:
    """Run WHY then WHAT (raw) for one customer and score Faithfulness on
    every free-text field. 2 generation calls + 1 embedding call + 2-5 judge
    calls -- real API usage, never call with a stub judge/client from a test."""
    catalog = catalog if catalog is not None else default_catalog()
    records: list[FaithfulnessRecord] = []

    def _sleep() -> None:
        if delay_between_calls_sec > 0:
            time.sleep(delay_between_calls_sec)

    profile = get_customer_profile(customer_id, source_path=source_path)
    insight = generate_customer_insight(
        customer_id, llm_client, source_path=source_path)
    _sleep()
    why_ctx = why_context(profile)
    why_score = _score_with_retry(insight.insight, why_ctx, judge_llm, question_for("insight"))
    _sleep()
    records.append(FaithfulnessRecord(
        stage="WHY", text_type="insight", customer_id=customer_id, faithfulness=why_score,
        response_text=insight.insight, contexts_json=_dump_contexts(why_ctx),
    ))

    ctx, nba = _generate_raw_nba(
        customer_id, llm_client, retriever, insight.insight, source_path, catalog,
    )
    _sleep()
    action_id = ctx.selection.recommended_action
    matched = next(
        a.matched_conditions for a in ctx.selection.evaluated_actions if a.action_id == action_id
    )
    explanation_ctx = what_explanation_context(matched, insight.insight)
    explanation_score = _score_with_retry(
        nba.explanation, explanation_ctx, judge_llm, question_for("explanation"))
    _sleep()
    records.append(FaithfulnessRecord(
        stage="WHAT", text_type="explanation", customer_id=customer_id,
        recommended_action=action_id, faithfulness=explanation_score,
        response_text=nba.explanation, contexts_json=_dump_contexts(
            explanation_ctx),
    ))

    for product in nba.products:
        known = catalog.get(product.product_id)
        if known is None:
            # Hallucinated product_id (not in catalog): nothing to score
            # Faithfulness against. rag/recommendation_validator.py already
            # flags this as a grounding failure; Faithfulness doesn't repeat it.
            continue
        reason_ctx = what_reason_context(known, ctx.risk_level)
        reason_score = _score_with_retry(product.reason, reason_ctx, judge_llm, question_for("reason"))
        _sleep()
        records.append(FaithfulnessRecord(
            stage="WHAT", text_type="reason", customer_id=customer_id,
            recommended_action=action_id, product_id=product.product_id, faithfulness=reason_score,
            response_text=product.reason, contexts_json=_dump_contexts(
                reason_ctx),
        ))

    return records


# -- checkpointed batch run ------------------------------------------------------

def _load_checkpoint(path: Path) -> dict[str, list[FaithfulnessRecord]]:
    if not path.exists():
        return {}
    done: dict[str, list[FaithfulnessRecord]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        import json

        row = json.loads(line)
        done.setdefault(row["customer_id"], []).append(
            FaithfulnessRecord(**row))
    return done


def run_evaluation(
    customer_ids: list[str],
    llm_client: LLMClient,
    judge_llm,
    retriever: ProductRetriever,
    source_path: Path = ALL_PATH,
    delay_between_calls_sec: float = 0.0,
    checkpoint_path: Optional[Path] = None,
) -> tuple[list[FaithfulnessRecord], RagasFaithfulnessSummary]:
    done = _load_checkpoint(checkpoint_path) if checkpoint_path else {}
    catalog = default_catalog()
    records: list[FaithfulnessRecord] = []

    for i, customer_id in enumerate(customer_ids):
        if customer_id in done:
            print(
                f"[{i + 1}/{len(customer_ids)}] {customer_id} already in checkpoint, skipping")
            records.extend(done[customer_id])
            continue
        print(
            f"[{i + 1}/{len(customer_ids)}] evaluating {customer_id}...", flush=True)
        customer_records = score_customer_faithfulness(
            customer_id, llm_client, judge_llm, retriever,
            source_path=source_path, catalog=catalog,
            delay_between_calls_sec=delay_between_calls_sec,
        )
        records.extend(customer_records)
        if checkpoint_path:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with checkpoint_path.open("a", encoding="utf-8") as f:
                for r in customer_records:
                    f.write(r.model_dump_json() + "\n")

    return records, compute_summary(records)


def rejudge_report(
    source_csv: Path,
    judge_llm,
    delay_between_calls_sec: float = 0.0,
) -> tuple[list[FaithfulnessRecord], RagasFaithfulnessSummary]:
    """Re-score Faithfulness from a report's stored texts and contexts, with a
    different judge. No generation, retrieval, or customer lookup -- only judge
    calls. Reports written before texts were stored cannot be re-judged."""
    df = pd.read_csv(source_csv, encoding="utf-8-sig")
    if "response_text" not in df.columns or "contexts_json" not in df.columns:
        raise ValueError(
            f"{source_csv} has no stored texts; regenerate it with the current script first")

    records: list[FaithfulnessRecord] = []
    for i, row in enumerate(df.to_dict("records")):
        if i > 0 and delay_between_calls_sec > 0:
            time.sleep(delay_between_calls_sec)
        clean = {k: (None if pd.isna(v) else v) for k, v in row.items()}
        if clean["response_text"] is None:
            score = None
        else:
            score = _score_with_retry(
                clean["response_text"], json.loads(clean["contexts_json"]), judge_llm,
                question_for(clean["text_type"]),
            )
        records.append(FaithfulnessRecord(**{**clean, "faithfulness": score}))

    return records, compute_summary(records)


# -- CLI -----------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    """Usage:
        python src/evaluation/ragas_evaluation.py                     # default small customer set
        python src/evaluation/ragas_evaluation.py --per-action 2      # larger (more quota)

    Needs GEMINI_API_KEY in .env (generation, embeddings, and the
    Faithfulness judge all use it). Resumable: rerun the same command after
    an interruption -- outputs/ragas_evaluation_checkpoint.jsonl is kept.
    """
    parser = argparse.ArgumentParser(
        description="Phase 6 RAGAS Faithfulness evaluation")
    parser.add_argument("--per-action", type=int, default=PER_ACTION_DEFAULT,
                        help="customers per selectable action (default: %(default)s)")
    parser.add_argument("--generation-model", default=None,
                        help="Gemini model for WHY/WHAT generation; overrides .env GEMINI_MODEL for this run only")
    parser.add_argument("--judge-model", default=None,
                        help="Gemini model for the Faithfulness judge; overrides .env RAGAS_JUDGE_MODEL for this run only")
    parser.add_argument("--rejudge-from", default=None,
                        help="re-score an existing report's stored texts with --judge-model; no generation")
    args = parser.parse_args(argv)

    load_env_file()

    if args.rejudge_from:
        judge_llm = get_faithfulness_llm(model=args.judge_model)
        records, summary = rejudge_report(
            Path(args.rejudge_from), judge_llm, delay_between_calls_sec=13.0)
        summary = summary.model_copy(
            update={"judge_model": resolve_judge_model(args.judge_model)})
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([r.model_dump() for r in records]).to_csv(
            REJUDGE_REPORT_PATH, index=False, encoding="utf-8-sig")
        REJUDGE_SUMMARY_PATH.write_text(
            summary.model_dump_json(indent=2), encoding="utf-8")
        print(
            f"\nRe-judged report -> {REJUDGE_REPORT_PATH}\nSummary -> {REJUDGE_SUMMARY_PATH}\n")
        print(summary.model_dump_json(indent=2))
        return

    llm_client = GeminiLLMClient(
        model=args.generation_model) if args.generation_model else select_llm_client()
    judge_llm = get_faithfulness_llm(model=args.judge_model)
    judge_model_name = resolve_judge_model(args.judge_model)
    generation_model_name = llm_client.model if isinstance(
        llm_client, GeminiLLMClient) else None

    from rag.retrieval import GeminiEmbeddingClient, load_or_build_collection, make_retriever

    embedding_client = GeminiEmbeddingClient()
    retriever = make_retriever(load_or_build_collection(
        embedding_client), embedding_client)

    # Gemini free tier: a few requests per minute: throttle between every real
    # call (generation, embedding, judge), not just between customers -- one
    # customer alone can make 5-7 real calls back to back.
    delay = 13.0 if isinstance(llm_client, GeminiLLMClient) else 0.0

    customer_ids = select_evaluation_customers(per_action=args.per_action)
    print(
        f"== RAGAS Faithfulness evaluation: {len(customer_ids)} customers ==")
    records, summary = run_evaluation(
        customer_ids, llm_client, judge_llm, retriever,
        delay_between_calls_sec=delay, checkpoint_path=CHECKPOINT_PATH,
    )
    CHECKPOINT_PATH.unlink(missing_ok=True)
    summary = summary.model_copy(update={
        "generation_model": generation_model_name,
        "judge_model": judge_model_name,
    })

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r.model_dump() for r in records]).to_csv(
        REPORT_PATH, index=False, encoding="utf-8-sig")
    SUMMARY_PATH.write_text(
        summary.model_dump_json(indent=2), encoding="utf-8")

    print(f"\nReport -> {REPORT_PATH}\nSummary -> {SUMMARY_PATH}\n")
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
