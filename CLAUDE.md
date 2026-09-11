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

The current project focuses on the WHO stage:
Priority Customer Ranking.
The system should identify customers who may have:
- high asset potential
- investment gap
- recent investment intent

The current implementation uses Rule-based Ranking as the baseline.
Do not introduce ML ranking in the current implementation.

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

The WHY layer must NOT rely on the LLM to discover customer facts directly.

Flow:

Structured Customer Data
    ↓
Evidence Extraction
    ↓
Key Evidence
    ↓
LLM
    ↓
Customer Insight

Key Evidence represents factual, traceable customer information.

Customer Insight is the natural-language interpretation generated from verified evidence.

Requirements:
- Customer facts must come from structured data.
- Every key evidence item must be traceable to source fields.
- LLM must not invent customer facts.
- LLM must not fabricate numbers, products, behaviors, or customer attributes.
- LLM should organize and explain evidence rather than independently inventing business facts.

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

The system must use a predefined Action Space.

Current candidate actions:

- A1: 投資需求訪談
- A2: 基金產品介紹
- A3: 定存到期資金規劃
- A4: 外幣資產配置討論
- A5: 既有投資組合檢視
- A6: 暫不聯繫

The AI provides decision support only.

It must NOT:
- execute transactions
- automatically contact customers
- make final suitability decisions
- autonomously determine actions outside the predefined Action Space

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
2. Evidence schema
3. Evidence extraction
4. Evidence validation
5. Customer Insight generation
6. LLM grounding validation
7. WHY evaluation

### Phase 3 — WHAT

1. Action Space definition
2. Business rule validation
3. Product Knowledge schema
4. Product Knowledge dataset
5. Retrieval
6. Reranking / filtering
7. Next Best Action generation
8. WHAT evaluation

### Phase 4 — Agent / Tool Use

1. Tool schema
2. Customer data tools
3. Product search tools
4. Agent workflow
5. Guardrails
6. End-to-end validation

### Phase 5 — Experiment / Optimization

1. Latency analysis
2. Retrieval experiments
3. Prompt / model experiments
4. Human evaluation
5. RM feedback

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
│   │    └── customers_scored.csv
├── src/
│   ├── ranking/
│   │    ├── generate_data.py
│   │    ├── feature_engineering.py
│   │    ├── customer_scoring.py
│   │    └── customer_priority.py
│   ├── rag/
│   ├── agent/
│   ├── tools/
│   └── evaluation/
├── outputs/
│   ├── priority_customers_all.csv
│   └── priority_customers_top10.csv
└── tests/
    ├── conftest.py
    ├── test_customer_priority.py
    ├── test_customer_scoring.py
    └── test_feature_engineering.py

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