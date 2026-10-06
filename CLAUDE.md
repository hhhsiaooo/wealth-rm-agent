# Wealth RM Agent

## Project Context

Build a Wealth Management RM decision-support AI Assistant.

The project consists of three decision-support tasks:

1. WHO: Priority Customer Ranking
2. WHY: Customer Evidence / Customer Insight
3. WHAT: Next Best Action

The overall product goal is to help Wealth Management RMs answer:

- WHO should I contact first?
- WHY is this customer worth contacting?
- WHAT action should I take next?

WHO, WHY, WHAT, the Agent, and a demo UI are all implemented (Phases 1-5
below); Phase 6 (Experiment / Optimization) is the only phase not started
beyond an initial latency log. WHO identifies customers who may have:
- high asset potential
- investment gap
- recent investment intent

WHO's implementation uses Rule-based Ranking as the baseline.
Do not introduce ML ranking.

More project information:
- docs/project_definition.md
- docs/ai_product_design.md

## Product Architecture

The product follows this decision-support flow:

Customer Data
    ↓
WHO: Priority Customer Ranking
    ↓
WHY: Customer Evidence / Customer Insight
    ↓
WHAT: Next Best Action
    ↓
RM Final Decision

Task responsibilities:

### WHO — Priority Customer

Purpose:
Identify which customers should be contacted first.

Current approach:
Rule-based Ranking.

Inputs:
Customer structured data and derived features.

Outputs:
- Priority Score
- Priority Rank
- Top-N priority customers
- Score contributions

Important:
WHO is a transparent business baseline.
Do not introduce ML ranking in the current implementation.

### WHY — Customer Evidence / Customer Insight

Purpose:
Explain why a customer is worth contacting.

Flow:
Structured Customer Data
    ↓
get_customer_profile(customer_id)
    ↓
Complete 43-field Customer Record
    ↓
LLM
    ↓
Relevant Fields + Customer Insight
    ↓
Fact Validation

The LLM may identify which fields are relevant to the question:
「為什麼這位客戶值得優先聯繫？」

Requirements:
- Customer facts must come from the structured customer record.
- The LLM may select relevant fields from the complete customer record.
- Every selected field and value must be traceable to the source record.
- LLM must not invent customer facts.
- LLM must not modify source values.
- LLM must not assume values that are not present.
- LLM must not create new business rules.
- LLM must not recalculate Priority Score.
- Fact validation must be performed after LLM generation.

Implementation Rules:
- Retrieve customer data through `get_customer_profile(customer_id)`.
- For a single-customer query, do not send the entire CSV to the LLM.
- The LLM may select relevant fields from the complete 43-field customer record.
- LLM output must use a structured schema.
- Validate LLM-selected fields and values against the source customer record.
- Do not introduce deterministic evidence extraction unless explicitly requested.
- Do not modify WHO ranking logic while implementing WHY.

### WHAT — Next Best Action

Purpose:
Recommend the next action an RM can take based on customer evidence and product knowledge.

Flow:
Customer Evidence
    +
Customer Conditions
    +
Product Knowledge
    ↓
Candidate Actions
    ↓
Eligibility / Business Rules
    ↓
Product Retrieval
    ↓
LLM
    ↓
Next Best Action

The system must use a predefined Action Space:
- A1: 投資需求訪談
- A2: 基金產品介紹
- A3: 定存資產配置討論
- A4: 外幣資產配置討論
- A5: 既有投資組合檢視
- A6: 暫不聯繫

Important:
- Do not create new business rules without explicit human approval.
- Do not allow LLM to determine eligibility rules.
- Eligibility / Business Rules must be deterministic and testable.
- Product Knowledge and Customer Data must remain separate.
- RAG is responsible for retrieving Product Knowledge.
- RAG must not retrieve or infer customer personal information.
- LLM must only generate explanations from validated customer information,
  business rules, and retrieved product context.
- Product recommendations must be traceable to Product Knowledge.
- The UI (src/ui/) only displays validated tool results; it must not call an LLM or re-implement business rules itself

## Development Workflow

The project must be implemented incrementally.

### Phase 1 — WHO

1. Specification validation
2. Raw synthetic data generation
3. Derived feature engineering
4. Feature scoring
5. Sub-score calculation
6. Priority Score
7. Ranking
8. Top-N output
9. WHO validation

### Phase 2 — WHY

1. WHY specification validation
2. Customer data tool
3. Customer record schema
4. LLM relevant-field selection
5. Customer Insight generation
6. Fact validation
7. LLM grounding validation
8. WHY evaluation

### Phase 3 — WHAT

1. WHAT specification validation
2. Action Space definition
3. Eligibility / Business Rules definition
4. Business rule validation
5. Product Knowledge schema
6. Synthetic Product Knowledge dataset
7. Product document preparation / chunking
8. Retrieval
9. Reranking / filtering
10. Next Best Action generation
11. Recommendation output validation
12. WHAT evaluation

### Phase 4 — Agent / Tool Use

1. Tool schema (three tools: Top-10 list / customer insight / next best action)
2. Tools (thin wrappers over the already-validated WHO / WHY / WHAT pipelines)
3. Agent workflow (autonomous tool-calling loop)
4. Guardrails
5. End-to-end validation

Phase 4 rules:
- The Agent orchestrates existing pipelines; it must not re-implement or bypass business rules, fact validation or recommendation validation.
- The LLM only decides which tool to call, asks for clarification, or says a request is unsupported. The final reply shown to the RM is rendered by code from tool results, never free-written by the LLM.
- get_next_best_action runs WHY first when no insight is cached for that customer in the session; if WHY is rejected, WHAT still runs without insight and says so.
- Any customer_id must exist in outputs/priority_customers_all.csv before a tool runs.
- LLM provider is swappable (Gemini default, Anthropic optional) behind an interface; tests use a stub LLM, never a real API.

### Phase 5 — UI

1. Chat interface over the Agent (Streamlit)
2. Sidebar priority list (WHO), no LLM call to render it
3. Structured display of tool results: status badges for the grounding
   outcome, expandable evidence fields, expandable tool-call trace
4. The UI only displays validated tool results; it must not call an LLM or
   re-implement business rules itself

### Phase 6 — Experiment / Optimization

1. Evaluation infrastructure -- in progress: RAGAS Faithfulness, LLM-as-judge
   rubric, Context Precision/Recall (docs/evaluation_design.md)
2. Latency analysis -- started: agent/latency.py logs per-stage latency
   (data retrieval, WHY/WHAT LLM generation, RAG retrieval, end-to-end) for
   every real Agent question to outputs/agent_latency_log.jsonl
3. Retrieval experiments
4. Prompt / model experiments
5. Human evaluation
6. RM feedback

Phase 6 rules:
- RAGAS metric calls (Faithfulness, any LLM-based Context Precision/Recall)
  must be stubbed in pytest, same as the existing stub-LLM rule -- never
  call a real provider from a test.
- RAGAS LLM-judge calls must reuse the existing Gemini free-tier throttling
  pattern (`delay = 13.0 if isinstance(llm_client, GeminiLLMClient) else 0.0`)
  used in why_evaluation.py/what_evaluation.py.

Do not skip phases.

Do not implement future phases unless explicitly requested.


## Project Structure

Do not create unnecessary files or directories.
Follow the existing project structure unless there is a clear reason
to change it.

wealth-rm-agent/
├── .gitignore
├── pyproject.toml
├── .venv/
├── CLAUDE.md
├── README.md
├── docs/
│   ├── project_definition.md
│   ├── ai_product_design.md
│   ├── evaluation_design.md
│   ├── experiment_design.md
│   └── ai_product_data_definition.xlsx
├── data/
│   └── synthetic/
│   │    ├── customers_raw.csv
│   │    ├── customers_features.csv
│   │    ├── customers_scored.csv
│   │    └── retrieval_golden_dataset.json
├── src/
│   ├── ranking/
│   │    ├── generate_data.py
│   │    ├── feature_engineering.py
│   │    ├── customer_scoring.py
│   │    └── customer_priority.py
│   ├── rag/
│   │    ├── schemas.py
│   │    ├── business_rules.py
│   │    ├── action_selector.py
│   │    ├── product_knowledge.py
│   │    ├── retrieval.py
│   │    ├── nba_generator.py
│   │    └── recommendation_validator.py
│   ├── agent/
│   │    ├── agent_llm.py
│   │    ├── agent_tools.py
│   │    ├── render.py
│   │    ├── rm_agent.py
│   │    └── latency.py
│   ├── ui/
│   │    ├── app.py
│   │    └── view_models.py
│   ├── insight/
│   │    ├── fact_validator.py
│   │    └── insight_generator.py
│   ├── tools/
│   │    └── customer_tool.py
│   └── evaluation/
│   │    ├── why_evaluation.py
│   │    ├── what_evaluation.py
│   │    ├── ragas_faithfulness.py
│   │    └── ragas_evaluation.py
├── outputs/
│   ├── priority_customers_all.csv
│   ├── priority_customers_top10.csv
│   ├── why_evaluation_report.csv
│   ├── why_evaluation_summary.json
│   ├── what_evaluation_report.csv
│   ├── what_evaluation_summary.json
│   ├── what_retrieval_report.csv
│   ├── what_retrieval_summary.json
│   ├── ragas_evaluation_report.csv
│   ├── ragas_evaluation_summary.json
│   ├── ragas_rejudge_report.csv
│   └── ragas_rejudge_summary.json
└── tests/
    ├── conftest.py
    ├── test_customer_priority.py
    ├── test_customer_scoring.py
    ├── test_customer_tool.py
    ├── test_fact_validator.py
    ├── test_feature_engineering.py
    ├── test_insight_generator.py
    ├── test_why_evaluation.py
    ├── test_business_rules.py
    ├── test_action_selector.py
    ├── test_product_knowledge.py
    ├── test_retrieval.py
    ├── test_nba_generator.py
    ├── test_recommendation_validator.py
    ├── test_what_evaluation.py
    ├── test_agent.py
    ├── test_agent_native.py
    ├── test_ui_view_models.py
    ├── test_ui_app.py
    ├── test_latency.py
    └── test_ragas_evaluation.py

## Data Specification

The Excel specification in:
docs/ai_product_data_definition.xlsx

is the source of truth for:
- data fields
- data types
- dependencies
- score mapping
- sub-score weights
- priority score weights

Do not invent additional scoring rules.

## Data Generation Workflow
The generation process should follow:

Step 1: Specification validation
Step 2: Raw synthetic data generation
Step 3: Derived feature engineering
Step 4: Feature scoring
Step 5: Sub-scores, priority score, ranking, and Top-N output

Do NOT implement the entire project in one step.
Complete one step at a time.

After each step:
- explain what was implemented
- show the relevant output
- identify potential issues
- Human-in-the-loop : wait for human review before proceeding

## Data Files

Raw:
data/synthetic/customers_raw.csv

Features:
data/synthetic/customers_features.csv

Feature Scores:
data/synthetic/customers_scored.csv

Ranking output:
outputs/priority_customers_all.csv
outputs/priority_customers_top10.csv

## Product Knowledge and RAG

Product Knowledge is used only in the WHAT stage.

Product Knowledge should contain structured product information such as:

- product_id
- product_category
- product_features
- risk_level
- target_customer
- investment_horizon
- FAQ
- sales_cautions
- effective_date
- source
- version

RAG responsibilities:

Customer Evidence
    ↓
Search Query
    ↓
Vector Retrieval
    ↓
Top-K Candidates
    ↓
Reranking / Filtering
    ↓
Product Context
    ↓
LLM
    ↓
Next Best Action Explanation

RAG is responsible for retrieving relevant product knowledge.

RAG must NOT:
- retrieve customer personal information
- replace structured customer data tools
- determine final customer suitability
- execute transactions

Structured customer data and product knowledge must remain separate.
- Customer Data→ Tool / structured query
- Product Knowledge→ RAG

## Evaluation Principles

Software testing and AI evaluation are separate.

### WHO

Current evaluation:
Business Logic Validation.

Verify that Top-N customers match the intended business hypotheses.

Do NOT claim:
- Precision@10
- Recall@10
- Conversion Rate
- A/B test results

unless independent real-world outcome labels are available.

Priority Score must never be used as ground truth.

### WHY

Evaluate:

- Customer Understanding Time
- Decision Time
- Factual Accuracy
- Evidence Coverage
- Evidence Traceability
- Hallucination Rate
- Faithfulness (free-text insight vs. source customer record)

Compare:
Customer Dashboard / Customer 360
vs.
AI-generated Customer Evidence

### WHAT

Evaluate:

- Product Retrieval Correctness
- Product Feature Accuracy
- Product Risk Accuracy
- Recommendation Consistency
- Business Rule Compliance
- Groundedness
- Faithfulness (free-text explanation/reason vs. matched conditions + retrieved product context)

### System Performance

Record separately:

- data retrieval latency
- product retrieval latency
- LLM generation latency
- end-to-end latency

Do not claim real business effectiveness without real customer interaction and outcome data.


## Coding Standards

Use Python.

Prefer:
- pandas for tabular data processing
- pathlib for file paths
- pytest for testing
- type hints where appropriate
- clear function names
- small, testable functions

Avoid:
- hard-coded business rules inside multiple functions
- duplicated scoring logic
- unnecessary frameworks
- overly complex architecture for this POC

## Package Management

This project uses Poetry for Python dependency management.

Rules:
- Do not use pip install for project dependencies.
- Use `poetry add <package>` to add runtime dependencies.
- Use `poetry add --group dev <package>` to add development dependencies.
- Use `poetry remove <package>` to remove dependencies.
- Use `poetry install` to install project dependencies.
- Use `poetry run <command>` to execute commands in the Poetry environment.
- Use `poetry run pytest` to run tests.
- Keep `poetry.lock` committed to Git.
- Before adding a new dependency, check whether the existing dependencies can satisfy the requirement.
- Do not add dependencies without explaining why they are needed.
- Do not manually edit dependency versions in `pyproject.toml` unless necessary.
- `ragas`, `langchain-core`, and `langchain-google-genai` are dev
  dependencies (used only by `src/evaluation/`, never by production
  `src/agent`/`src/rag`/`src/insight` code) -- add via
  `poetry add --group dev "ragas>=0.3.9,<0.4.0" langchain-core langchain-google-genai`.
  Every currently-available ragas release unconditionally imports
  `langchain_community.chat_models.vertexai.ChatVertexAI` from
  `ragas/llms/base.py`, a path removed from langchain-community in its
  0.4.2 release -- `import ragas` crashes regardless of ragas version or
  provider unless that module is present. Rather than pin an old
  langchain-community (risking a conflict with the langchain-core version
  already in use), `src/evaluation/ragas_faithfulness.py` stubs the dead
  import path in `sys.modules` before `ragas` is imported. Delete that stub
  once ragas ships the real upstream fix.
  Gemini is the only LLM judge wired up for now (no Anthropic API plan);
  add `langchain-anthropic` later if an Anthropic judge is needed.

## Important Rules

- Do not calculate derived fields during raw data generation.
- Do not independently randomize fields that have dependencies.
- Keep raw data separate from derived features.
- Keep feature scoring separate from sub-score calculation.
- Do not recalculate feature scores in the priority calculation stage.
- Do not introduce ML.
- Do not use the final Priority Score as ground truth.
- Use a fixed random seed for reproducibility.
- Preserve deterministic outputs.
- Add validation and pytest tests for important boundaries and business rules.
- Do not modify or replace existing rule-based validation
  (`insight/fact_validator.py`, `rag/recommendation_validator.py`) or
  existing evaluation summary fields when adding RAGAS/LLM-judge metrics.
  New metrics are independent additions only (see docs/evaluation_design.md).

## Collaboration Rules

Before coding:
1. Read the relevant specification.
2. Explain the planned implementation.
3. Identify assumptions or ambiguities.
4. Ask for clarification only if necessary.

After coding:
1. Explain what changed.
2. List modified files.
3. Run relevant tests.
4. Report test results.
5. Do not silently change business logic.