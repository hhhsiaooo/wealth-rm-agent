# 財富管理客戶經營決策支援 AI Assistant

## 一、Product Overview
### Product Goal
將現有理專客戶經營儀表板 **從「提供資訊」進一步提升為「提供決策支援」**，協助理專在電話聯繫客戶前快速釐清：今天應該優先聯繫誰？為什麼值得聯繫？聯繫時可以採取什麼行動？
| 項目 | 任務 | 核心技術 | 輸出 |
| :---: | :---: | :---: | :---: |
| WHO | Priority Customer | Rule-based Ranking | Top 10 |
| WHY | Customer Evidence | Structured Data + LLM | Customer Insight |
| WHAT | Next Best Action | RAG + LLM | Recommended Action |
### Scope
* Priority Customer Ranking
* Customer Evidence Generation
* Customer Insight Generation
* Next Best Action Suggestion
* Product Knowledge RAG
* Function Calling / Tool Use
* LLM Explanation
* AI Evaluation
* Experiment / Optimization
### Out of Scope
* Automated Customer Contact
* Automated Transactions
* Real Customer Data
* Automated Data Ingestion
* Automatic Score Updates

## 二、Product Workflow
```
Data Definition
↓
Score Definition
↓
Task 1: WHO — Priority Customer
↓
Task 2: WHY — Customer Evidence
↓
Task 2: Customer Insight
↓
Task 3: WHAT — Next Best Action
↓
Product Knowledge RAG
↓
Business Rules
↓
RM Final Decision
```

## 三、Data Definition
* 欄位定義、資料型態與計算方式請參考`docs/ai_product_data_definition.xlsx`。
* 本專案使用一次性生成的 Synthetic Customer Data。資料自動更新與排程不在本次 POC 範圍。
### Data Specification
定義：、Raw Data、Derived Feature
**1. 客戶基本資料**
* `customer_id` 客戶唯一識別碼
* `occupation` 職業
* `job_level` 職稱
* `gender` 性別
* `age` 年齡
* `risk_level` 風險屬性
**2. 產品持有情形**
* `has_twd_td` 是否持有台幣定存
* `has_fx_deposit` 是否持有外幣存款
* `has_fund` 是否持有基金
* `has_robo` 是否持有機器人理財
* `has_bond` 是否持有債券
* `has_insurance` 是否持有保險
**3. 資產結構**
* `twd_demand_balance` 台幣活存餘額
* `twd_td_balance` 台幣定存餘額
* `fx_demand_balance` 外幣活存餘額
* `fx_td_balance` 外幣定存餘額
* `investment_balance` 理財庫存餘額：
* `total_aum` 管理資產總額 AUM：`twd_demand_balance` + `twd_td_balance` + `fx_demand_balance` + `fx_td_balance` + `investment_balance`
* `twd_demand_ratio` 台幣活存佔比：`twd_demand_balance` / `total_aum`
* `td_ratio` 定存佔比：(`twd_td_balance` + `fx_td_balance`) / `total_aum`
* `fx_ratio` 外幣佔比：(`fx_demand_balance` + `fx_td_balance`) / `total_aum`
* `investment_ratio` 理財庫存佔比：`investment_balance` / `total_aum`
**4. 資金流**
* `inflow_1m` 近 1 個月轉入金額
* `outflow_1m` 近 1 個月轉出金額
**5. 基金申購行為**
* `fund_buy_count_1m` 近 1 個月申購基金次數
* `fund_buy_amount_1m` 近 1 個月申購基金金額
* `has_rm` 是否有理專
**6. 數位行為**
* `fund_view_count_1m` 近 1 個月基金頁瀏覽次數
* `has_abandoned_application_1m` 是否近 1 個月未完成基金申購
### Score Transformation
基於Score Feature、Feature Dependencies，將做為排序依據的 Feature 轉換為 0–100 分。

## 四、Scoring Definition
### Asset Potential Score
資產潛力高 = AUM 高 + 職業 + 職稱

| Feature | Weight |
| :---: | :---: |
| total_aum_score | 70% |
| occupation_score | 15% |
| job_level_score | 15% |

### Investment Gap Score
有閒置資金與配置缺口 = 現金比例高 + AUM 高 + 投資比例低 + 沒有基金

| Feature | Weight |
| :---: | :---: |
| twd_demand_ratio_score | 40% |
| total_aum_score | 30% |
| investment_ratio_score | 20% |
| has_fund_score | 10% |

### Investment Intent Score
有投資意圖 = 近期轉入大額資金 + 未完成申購 + 多次瀏覽

| Feature | Weight |
| :---: | :---: |
| inflow_1m_score | 40% |
| has_abandoned_application_1m_score | 35% |
| fund_view_count_1m_score | 25% |

### Priority Score
資產潛力高 + 有閒置資金或配置缺口 + 有投資意圖

| Score | Weight |
| :---: | :---: |
| Investment Gap Score | 45% |
| Investment Intent Score| 35% |
| Asset Potential Score| 20% |

## 五、Task
### 1. Priority Customer
#### Business Question
如何在 1,000 位財管客戶中，找出最值得優先聯繫的 Top 10 客戶。
```
All 1,000 Customer Data > Scoring > Priority Score > Top 10 Customer Data
```
#### Input
欄位定義與內容請參考「三、Data Definition 」。
* Raw Data(24 個欄位)：`data/synthetic/customers_raw.csv`。
* 加上 Derived Features (5 個欄位)：`data/synthetic/customers_features.csv`。
* 加上 Feature Scores (9 個欄位)：`data/synthetic/customers_scored.csv`。
#### Output
根據資產潛力、有閒置資金或配置缺口、有投資意圖等訊號計算 Priority Score。
* 加上 SubScores (3 個欄位)、Priority Score (1 個欄位)、Rank (1 個欄位)：`outputs/priority_customers_all.csv`。共 1,000 筆客戶資料，43 個欄位 。
* 篩選排序前 10 的客戶資料：`outputs/priority_customers_top10.csv`。
#### Constraint
僅使用取得且經授權的客戶資料，每個 Priority Score 必須可解釋。
#### Evaluation
Contact Rate、Conversion Rate、Precision@10、Recall@10。
#### Experiment
Rule-based Ranking（Baseline） vs AI/ML Ranking

### 2. Customer Evidence
#### Business Question
將客戶多維度資料轉換成 RM 可以快速理解的客戶洞察與聯繫理由。
```
LLM > Tool calling：get_customer_profile(customer_id) > Customer Record > LLM > Customer Insight + Relevant Fields > Fact Validation
```
#### Input
透過 Function/Tool 取得指定客戶資料。
程式碼：`tools/customer_tool.py`。
* Function：`get_customer_profile(customer_id)`
* Resource：`outputs/priority_customers_top10.csv`
* Input：`customer_id: string`
* Output：JSON object containing the customer's 43 fields.
* 系統不會將完整的 `priority_customers_top10.csv`
傳給 LLM，而是先根據 `customer_id` 取得單一客戶的 43-field Customer Record，再提供給 LLM。
* Example:
```
{
"customer_id": "C0001",
"occupation": "finance",
"job_level": "Manager",
...
"total_aum": 8000000,
"twd_demand_ratio": 0.72,
"investment_ratio": 0.08,
...
"asset_potential_score": 85,
"investment_gap_score": 92,
"investment_intent_score": 88,
"priority_score": 90,
"priority_rank": 1
}
```
#### Output
取得單一客戶的 43-field Customer Record ，根據客戶排序所使用的訊號，由 LLM 組織成 Customer Insight + Relevant Fields。
程式碼：`insight/insight_generator.py`。

LLM receives the complete structured customer record
returned by `get_customer_profile(customer_id)`.
LLM should:
1. Identify customer fields relevant to the question:「為什麼這位客戶值得優先聯繫？」
2. Identify the major signals contributing to the customer's Priority Score.
3. Combine relevant customer facts into a concise Customer Insight.
4. Explain why these facts make the customer worth RM attention.
5. Return the selected fields used to support the insight.

LLM must NOT:
1. Invent customer facts.
2. Modify customer values.
3. Assume values that are not present.
4. Create new business rules.
5. Recalculate Priority Score.
6. Recommend a financial product at the WHY stage.
7. Make a final suitability judgment.

Example:
```
{
  "customer_id": "C0001",
  "priority_rank": 1,
  "priority_score": 90,
  "key_fields": [
    {
      "field": "total_aum",
      "value": 8000000
    },
    {
      "field": "twd_demand_ratio",
      "value": 0.72
    },
    {
      "field": "investment_ratio",
      "value": 0.08
    },
    {
      "field": "has_fund",
      "value": 0
    },
    {
      "field": "fund_view_count_1m",
      "value": 18
    }
  ],
  "insight": "客戶目前資產約800萬元，台幣活存占AUM約72%，理財庫存占比僅8%，目前沒有基金持有；近1個月有18次基金瀏覽，因此可優先了解其投資需求。"
}
```

Pydantic model:
```
class KeyField(BaseModel):
    field: str
    value: Any

class CustomerInsight(BaseModel):
    customer_id: str
    priority_rank: int
    priority_score: float
    key_fields: list[KeyField]
    insight: str
```
#### Constraint
LLM 不得產生不存在於 Customer Data 的事實，必須可追溯至資料來源。

#### Fact Validation
驗證 LLM 產出的 Customer Insight + Relevant Fields 是否與 Customer Data Resource 相符。
程式碼：`insight/fact_validator.py`。

After LLM generation, the system validates every key field returned by the LLM against the original 43-field customer record retrieved from the WHO Ranking Output.

The source dataset is:
`outputs/priority_customers_top10.csv`

This file contains the Top 10 priority customers
(or more than 10 customers when tied ranks occur),
with 43 fields per customer.
When a Customer ID is provided, the system retrieves only the corresponding customer record instead of sending the entire CSV to the LLM.

Validation should check:
* `customer_id` exists and matches the requested customer
* selected field exists in the source record
* selected value matches the source value
* `priority_score` matches the source value
* `priority_rank` matches the source value
If validation fails, the response should not be treated as fully grounded.

#### Evaluation
Decision Time、Factual Accuracy、Hallucination Rate
#### Experiment
Raw Customer 360 Dashboard vs AI-generated Customer Evidence
降低 RM 理解客戶所需時間，同時維持高事實正確率與低 Hallucination。

### 3. Next Best Action
#### Business Question
根據 Customer Evidence 與產品資訊，提供 RM 可採取的下一步行動建議 Decision Support
#### Input
Customer Evidence → Candidate Actions → Eligibility / Business Rules → Recommended Action
#### Output
先定義 AI 可以選什麼。
Action	Description
A1	投資需求訪談
A2	基金產品介紹
A3	定存到期資金規劃
A4	外幣資產配置討論
A5	既有投資組合檢視
A6	暫不聯繫
#### Constraint
僅能從預先定義的 Action Space 中產生建議；不得直接執行交易或取代 RM 的最終適合度判斷
#### Evaluation
Decision Time、Action Acceptance、Task Completion、Conversion Rate
Recommendation Accuracy、Action Agreement、Groundedness、Hallucination Rate
#### Experiment
RM 根據經驗自行決定聯繫策略 vs Customer Evidence + NBA
提升 RM 決策效率與行動一致性，且推薦內容具有資料與產品知識依據

## 六、Function / Tool Definition
* 取得指定客戶資料 `get_customer_profile(customer_id)`
  * Purpose：取得指定客戶的完整 Structured Customer Data。
  * Input：customer_id: string
  * Output：JSON object containing the customer's 43 fields from WHO Ranking Output.
  * Example:
```
{
"customer_id": "C0001",
"occupation": "finance",
"job_level": "Manager",
...
"total_aum": 8000000,
"twd_demand_ratio": 0.72,
"investment_ratio": 0.08,
...
"asset_potential_score": 85,
"investment_gap_score": 92,
"investment_intent_score": 88,
"priority_score": 90,
"priority_rank": 1
}
```

## 七、RAG
當 AI 要回答「這個產品是什麼、適合什麼樣的客戶、有哪些注意事項」時，知識從哪裡來？
RAG is used to retrieve up-to-date product knowledge and provide grounded context for AI-generated explanations and candidate action recommendations.
### Knowledge Base
Structure
```
Product Knowledge
├── Fund A
├── Fund B
├── Fund C
├── Insurance A
├── FX Product A
└── FAQ
```
每份文件切 Chunk 後加入 metadata
```
product_id
product_category
risk_level
target_customer
investment_horizon
effective_date
source
version
```
### RAG Flow
```
Customer Evidence
        ↓
Generate Search Query
        ↓
Vector Search
        ↓
Top-K Documents
        ↓
Reranking / Filtering
        ↓
Relevant Product Context
        ↓
LLM
        ↓
Grounded Response
```
### Output
不要只有：「推薦基金 A。」
而是：
「根據產品資料，基金 A 的風險屬性為中高風險，主要投資於 XXX，適合投資期間 XXX。由於客戶目前風險屬性為中高風險，因此可作為後續需求討論的候選產品，但仍需進一步確認適合度。」
這樣才能測：Retrieval Quality、Groundedness、Hallucination、Recommendation Accuracy

## 八、Agent Workflow
```
                    RM
                     │
                     ▼
             User Intent
                     │
                     ▼
              Agent Planner
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
  Customer Data Tools       Product RAG
          │                     │
          ▼                     ▼
   Customer Evidence      Product Knowledge
          │                     │
          └──────────┬──────────┘
                     ▼
              Decision Logic
                     │
                     ▼
             Next Best Action
                     │
                     ▼
             LLM Explanation
                     │
                     ▼
              Final Response
```
## Agent Responsibilities
Component	Responsibility
Database / SQL	提供客戶事實
Ranking Model / Rule	找 Priority Customer
Tool	取得資料
RAG	找產品知識
Business Rules	Eligibility / Constraint
LLM	整合、推理、解釋
Agent	Orchestration

AI / Non-AI Responsibility

