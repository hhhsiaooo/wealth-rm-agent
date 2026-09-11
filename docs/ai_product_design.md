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
資產潛力高 + 有閒置資金與配置缺口 + 有投資意圖

| Score | Weight |
| :---: | :---: |
| Investment Gap Score | 45% |
| Investment Intent Score| 35% |
| Asset Potential Score| 20% |

## 五、Task 1 — WHO: Priority Customer


## 三、Product Knowledge
* 來源：Documents → RAG
* 資料：
Product Description
Fund Prospectus
FAQ
Risk Information
Product Features
Sales Guidelines

## 四、Task
### 1. Priority Customer
#### Business Question
在目前約 2,000 位財管客戶中，如何找出最值得優先聯繫的 Top 10 客戶。
#### Input
根據資產規模、投資配置缺口、近期投資意圖等訊號計算 Priority Score。
#### Output
Top 10 Priority Customers
Rank	Customer	Score	Key Evidence
1	C001	92	AUM 高、現金占比高、近期瀏覽基金
2	C017	87	資金近期增加、目前基金持有低
3	C032	84	高資產、近期有投資行為
#### Constraint
僅使用取得且經授權的客戶資料；每個 Priority Score 必須可解釋。
#### Evaluation
Contact Rate、Qualified Contact Rate、Conversion Rate
Precision@10、Recall@10（視 Golden Dataset 定義）
#### Experiment
Rule-based Ranking vs AI/ML Ranking（MVP 可先以 Rule-based 作為 Baseline）
Score=AUM Score+ Cash Allocation Score+ Investment Gap Score+ Recent Intent Score+ Behavior Score
AI Ranking 在 Precision@10 等離線指標優於 Baseline，且不增加過多決策時間

### 2. Customer Evidence
#### Business Question
將客戶分散的多維度資料轉換成 RM 可以快速理解的客戶洞察與聯繫理由
#### Input
根據 Priority Ranking 所使用的訊號，擷取相關 Customer Facts，再由 LLM 組織成 Evidence
#### Output
Customer：C001
Priority：High
Key Evidence：
* AUM 約 800 萬
* 台幣存款占資產約 72%
* 目前無基金持有
* 近三個月有 8 次基金瀏覽
* 近期有未完成申購紀錄
Business Interpretation：客戶目前資產規模較高，但投資型產品配置較低，近期又出現基金相關行為，因此具有進一步了解投資需求的訊號。
Data Sources：Asset / Product / Digital Behavior
#### Constraint
LLM 不得產生不存在於 Customer Data 的事實；Evidence 必須可追溯至資料來源
#### Evaluation
Customer Understanding Time、Decision Time
Factual Accuracy、Evidence Coverage、Faithfulness、Hallucination Rate
#### Experiment
Raw Customer 360 vs AI-generated Customer Evidence
降低 RM 理解客戶所需時間，同時維持高事實正確率與低 Hallucination

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

### 三、Tool
AI Agent 不應該直接碰 Database。
不要全部丟進 Vector DB。
Structured Data → Tool / SQL
Knowledge → RAG
Reasoning / Explanation → LLM
* `get_customer_profile(customer_id)`
查客戶基本資料
* `get_customer_assets(customer_id)`
查資產
* `get_customer_behavior(customer_id)`
查行為
* `search_priority_customers(criteria)`
找候選客戶
* `search_products(criteria)`
搜尋產品
* `get_product_detail(product_id)`
取得產品資訊

### 四、RAG
當 AI 要回答「這個產品是什麼、適合什麼樣的客戶、有哪些注意事項」時，知識從哪裡來？
#### Purpose
RAG is used to retrieve up-to-date product knowledge and provide grounded context for AI-generated explanations and candidate action recommendations.
#### Knowledge Base
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
#### RAG Flow
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
#### Output
不要只有：「推薦基金 A。」
而是：
「根據產品資料，基金 A 的風險屬性為中高風險，主要投資於 XXX，適合投資期間 XXX。由於客戶目前風險屬性為中高風險，因此可作為後續需求討論的候選產品，但仍需進一步確認適合度。」
這樣才能測：Retrieval Quality、Groundedness、Hallucination、Recommendation Accuracy

### 五、Agent Workflow
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
#### Agent Responsibilities
Component	Responsibility
Database / SQL	提供客戶事實
Ranking Model / Rule	找 Priority Customer
Tool	取得資料
RAG	找產品知識
Business Rules	Eligibility / Constraint
LLM	整合、推理、解釋
Agent	Orchestration

AI / Non-AI Responsibility

