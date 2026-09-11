### Business Problem
### Existing Dashboard
### Decision Gap
### AI Solution
### Architecture
### Demo
logging
retrieval_latency
tool_latency
llm_latency
total_latency
### Evaluation
### Experiment
### Results
### Limitations
### Future Work

```
wealth-management-ai/
│
├── README.md
│
├── docs/
│   ├── project_definition.md
│   ├── ai_product_design.md
│   ├── experiment_design.md
│   └── evaluation.md
│
├── data/
│   ├── customers.csv
│   ├── products/
│   └── golden_dataset.json
│
├── src/
│   │
│   ├── data/
│   │   ├── generator.py
│   │   └── repository.py
│   │
│   ├── ranking/
│   │   ├── rule_based.py
│   │   └── priority_service.py
│   │
│   ├── tools/
│   │   ├── customer_tools.py
│   │   └── product_tools.py
│   │
│   ├── rag/
│   │   ├── ingest.py
│   │   ├── retriever.py
│   │   └── evaluation.py
│   │
│   ├── agent/
│   │   ├── prompts.py
│   │   ├── workflow.py
│   │   └── decision_agent.py
│   │
│   ├── evaluation/
│   │   ├── ranking_eval.py
│   │   ├── rag_eval.py
│   │   └── latency_eval.py
│   │
│   └── observability/
│       └── logger.py
│
├── tests/
│
└── app.py
```