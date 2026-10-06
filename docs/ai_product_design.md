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
        Score Definition
                │
                ▼
    Task 1: WHO — Priority Customer
          規則計分，不呼叫 LLM
   輸出：priority_customers_all.csv／top10.csv
                │
                ▼
      RM 開啟網頁看到今日優先名單
                │
                ▼
      RM 用自然語言提問
                │
                ▼
      Agent（LLM）判斷要呼叫哪個工具
                │
        ┌───────┴───────┐
        ▼               ▼
 Task 2: WHY          Task 3: WHAT
 get_customer_profile   規則層選行動（ Business Rules）
      ↓                     ↓
  LLM 生成洞察        行動 A2/A3/A4 觸發 Product Knowledge 
      ↓                     ↓
  Fact Validation      LLM 生成說明＋挑選產品
  （查核／必要時重試）        ↓
                        Recommendation Validation
                        （查核／必要時重試）
        └───────┬───────┘
                ▼
      程式碼組成回覆（非 LLM 生成文字）
                │
                ▼
       理專看到查核狀態、依據欄位、候選產品
                │
                ▼
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
* `investment_balance` 理財庫存餘額
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
每個 Priority Score 必須可解釋。
#### Evaluation
Business Logic Validation：檢查 Top 10 是否符合預期業務假設（如 AUM 越高分數越高、投資占比越低缺口越高）
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
* Resource：可查詢 `priority_customers_all.csv`（任意客戶）或 `priority_customers_top10.csv`（今日優先名單）
* Input：`customer_id: string`
* Output：JSON object containing the customer's 43 fields.
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

LLM receives the complete structured customer record returned by `get_customer_profile(customer_id)`.

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
`outputs/priority_customers_all.csv`（任意客戶）
`outputs/priority_customers_top10.csv`（今日優先名單）

When a Customer ID is provided, the system retrieves only the corresponding customer record instead of sending the entire CSV to the LLM.

Validation should check:
* `customer_id` exists and matches the requested customer
* selected field exists in the source record
* selected value matches the source value
* `priority_score` matches the source value
* `priority_rank` matches the source value
If validation fails, the response should not be treated as fully grounded.

Pydantic model:
```
IssueType = Literal[
    "customer_id_mismatch",
    "unknown_field",             # 幻覺：欄位在來源資料裡根本不存在
    "value_mismatch",            # 數值錯誤：欄位存在，但值不符
    "priority_score_mismatch",
    "priority_rank_mismatch",
]

class GroundingIssue(BaseModel):
    type: IssueType
    detail: str

class FactCheckResult(BaseModel):
    customer_id: str
    is_grounded: bool
    issues: list[GroundingIssue]
```

#### Grounding Policy
* 查核失敗處理方式：
  * 生成 → 查核。
  * 未通過，再查核 → 把「具體錯誤清單」附在 prompt 後重試一次。
  * 仍未通過，拒絕 → `status = "rejected"`，不顯示 LLM 洞察，改顯示固定訊息「AI 洞察未通過事實查核，請直接查閱客戶原始資料。」
  * 每一次嘗試（不論成敗）都追加一行到 `outputs/grounding_audit_log.jsonl` 供人工稽核。
* 程式碼：`insight/fact_validator.py` 的 `generate_grounded_customer_insight()`，結果以 `GroundingOutcome` 回傳（`status`：`grounded`／`grounded_after_retry`／`rejected`）。
* Pydantic 驗證：`insight/insight_generator.py`（輸出層：`KeyField`、`CustomerInsight`）、`insight/fact_validator.py`（查核層：`GroundingIssue`、`FactCheckResult`）。

#### Evaluation
程式碼：`evaluation/why_evaluation.py`。
* Factual Accuracy（raw）：第一次 LLM 輸出即通過查核的客戶比例（衡量 LLM／prompt 本身品質）。
* Factual Accuracy（final）：經 Grounding Policy 之後通過的比例（理專實際看到的結果）。
* Hallucination Rate：首次輸出的 `key_fields` 中，欄位名稱在來源資料根本不存在的比例。
* Value Mismatch Rate：首次輸出的 `key_fields` 中，欄位存在但值不符的比例。
* Decision Time：與 Customer 360 人工方式做對照（目前記錄`mean_data_retrieval_latency_sec`: 0.01  `mean_end_to_end_latency_sec`: 9.47）
* 輸出：`outputs/why_evaluation_report.csv`（TOP10客戶）、`outputs/why_evaluation_summary.json`。

#### Experiment
Raw Customer 360 Dashboard vs AI-generated Customer Evidence
降低 RM 理解客戶所需時間，同時維持高事實正確率與低幻覺。


### 3. Next Best Action
#### Business Question
根據 Customer Evidence 與產品資訊，提供 RM 可採取的下一步行動建議。
```
Customer Evidence → Candidate Actions → Eligibility / Business Rules → Action Selection → Product Knowledge Retrieval → Recommended Action → LLM Explanation → RM Final Decision
```
#### Input
* **Customer Evidence**
  * **Relevant Fields**：行動判斷欄位。
  * **Customer Insight**：撰寫行動建議的參考文字。
* 若 WHY 尚未執行或被拒絕，WHAT 仍可執行，prompt 會註明「沒有可用的客戶洞察」。
* 客戶資料來源： `priority_customers_all.csv`（任意客戶）或 `priority_customers_top10.csv`。

* **Candidate Actions**

| Action ID | Action | Description |
|---|---|---|
| A1 | 投資需求訪談 | 了解客戶投資目標、需求與風險承受度 |
| A2 | 基金產品介紹 | 根據客戶條件檢索符合條件的基金產品 |
| A3 | 定存資產配置討論/產品介紹 | 討論定存資產配置需求 |
| A4 | 外幣資產配置討論/產品介紹 | 討論外幣資產配置需求 |
| A5 | 既有投資組合檢視 | 檢視既有投資組合與配置情況 |
| A6 | 暫不聯繫 | 當目前沒有明確聯繫訊號時，不建議主動聯繫 |

* **Eligibility / Business Rules**
程式碼：`rag/business_rules.py`，Python 規則判斷，不呼叫 LLM
  * **A1 投資需求訪談**
  investment_intent_score >= 50 
  AND investment_ratio < 0.30
  * **A2 基金產品介紹**
  has_fund = 0 
  AND investment_intent_score >= 70 
  AND risk_level IN {Medium, High}
  * **A3 定存資產配置討論/產品介紹**
  twd_td_balance > 0 
  OR fx_td_balance > 0
  * **A4 外幣資產配置討論/產品介紹**
  fx_demand_balance > 0
  OR fx_td_balance > 0
  * **A5 既有投資組合檢視**
  investment_balance > 0 
  AND investment_ratio >= 0.30
  * **A6 暫不聯繫**
  不符合 A1～A5 任一條件
  * 產品 `risk_level` 不得高於客戶 `risk_level`（Low < Medium < High）。Low 客戶只能看到 Low 產品；Medium 客戶看到 Low、Medium；High 客戶看到 Low、Medium、High。

* **Action Selection**
程式碼：`rag/action_selector.py`，Python 規則判斷，不呼叫 LLM。
  ```
  A2 > A3 > A4 > A5 > A1 > A6
  ```

  Pydantic model:
  ```
  class ActionEligibility(BaseModel):
      action_id: str                  # "A1".."A6"
      eligible: bool
      matched_conditions: list[str]   # 程式碼直接產生，例如 ["investment_intent_score=92 >= 70", "has_fund=0"]

  class ActionSelectionResult(BaseModel):
      customer_id: str
      evaluated_actions: list[ActionEligibility]   # 六條規則的完整判斷結果，供稽核
      recommended_action: str                       # 套用優先順序後選出的唯一答案
  ```
* **Product Knowledge**
  * 產品基本資訊
  * 產品風險屬性
  * 適用客群
  * 投資期間
  * 注意事項

#### Output
程式碼：
* `rag/schemas.py`：Pydantic 資料模型。
* `rag/business_rules.py`、`rag/action_selector.py`：取得客戶欄位資料，用規則判斷層選出唯一推薦行動，不呼叫 LLM。
* `rag/product_knowledge.py`、`rag/retrieval.py`：Product Knowledge Retrieval（RAG）僅在 `recommended_action` 為 A2／A3／A4 時觸發（對應 Fund／TD／FX 三個 Knowledge Base 分類）。A1／A5／A6 仍需要 LLM 產生自然語言說明（不含產品檢索）。
* `rag/nba_generator.py`：串接整條 WHAT 流程。LLM 不能自己判斷行動，只能決定 `explanation` 怎麼寫、以及 `products` 要引用哪些 retrieved 內容。
* `rag/recommendation_validator.py`：Recommendation Validation

LLM should:
1. 根據已通過 Eligibility / Business Rules 的 Action，組織推薦行動說明。
2. 根據 RAG retrieved context 產生自然語言說明。
3. 引用 retrieved product information 作為推薦依據。

LLM must NOT:
1. Create new Candidate Actions.
2. Create new Eligibility / Business Rules.
3. Ignore defined Business Rules.
4. Invent product information.
5. Modify product information.
6. Recommend products that are not present in retrieved Product Knowledge.
7. Make final suitability judgments.
8. Execute transactions.
9. Contact customers automatically.

Next Best Action must be returned using a structured schema. Product information must be traceable to retrieved Product Knowledge.

**搜尋查詢怎麼產生**
程式碼：`rag/nba_generator.py`
搜尋查詢由程式碼依客戶欄位以固定樣板產生（不由 LLM 產生），只使用能對應到產品屬性的條件：
* 一律帶入：客戶 `risk_level`（低／中／高風險）+ 動作對應的需求語（A2「適合投資需求」、A3「適合資金停泊與定存配置」、A4「適合外幣資產配置」）+ 產品類別。
* 僅 A3：若客戶只持有台幣定存，加「台幣」；只持有外幣定存，加「外幣」；兩者都有則不加。
* 例：`高風險、適合投資需求的基金產品`。
* WHY 的 Customer Insight 不放進查詢，只用來撰寫建議說明。
* 因此客戶條件的影響分三層：規則層依欄位數值決定行動；檢索層依風險屬性與（A3）幣別縮小候選產品；LLM 從檢索結果中挑選產品並撰寫說明。已知限制：同一風險等級、同一行動的客戶會檢索到相同的候選產品，差異只在 LLM 的挑選與說明。

Required fields:
* customer_id
* recommended_action
* action_name
* eligible
* explanation
* products
* sources

Example:
```
{
  "customer_id": "C0001",
  "recommended_action": "A2",
  "action_name": "基金產品介紹",
  "eligible": true,
  "explanation": "客戶近1個月有多次基金瀏覽，且目前沒有基金持有。",
  "products": [
    {
      "product_id": "FUND_A",
      "product_name": "基金 A",
      "reason": "產品風險屬性與客戶風險屬性符合，投資期間亦符合產品條件。"
    }
  ],
  "sources": [
    "product_knowledge/FUND_A"
  ]
}
```

Pydantic model:
```
class ProductRecommendation(BaseModel):
    product_id: str
    product_name: str
    reason: str

# LLM 只能輸出這兩個欄位
class NBALLMOutput(BaseModel):
    explanation: str
    products: list[ProductRecommendation]

# 程式碼組裝
class NextBestAction(BaseModel):
    customer_id: str
    recommended_action: str      # 複製自規則判斷層
    action_name: str             # 複製自規則判斷層
    eligible: bool             # 複製自規則判斷層
    explanation: str        
      # LLM 生成的自然語言說明
    products: list[ProductRecommendation] = [] 
      # LLM 從檢索結果中挑選產品並撰寫說明
    sources: list[str] = []       
      # "product_knowledge/{product_id}"
```
#### Constraint
僅能從預先定義的行動清單中產生建議；不得直接執行交易或取代 RM 的最終適合度判斷。

#### Recommendation Validation
程式碼：`rag/recommendation_validator.py/validate_next_best_action()`。

驗證 LLM 產出的 `NextBestAction` -> (`rag/nba_generator.py/assemble_next_best_aaction()`)
是否與規則判斷層 `ActionSelectionResult` -> (`rag/action_selector.py/select_action_for_profile()`)
以及 retrieved Product Knowledge 相符，若驗證失敗，不得視為可信賴的推薦內容。

Validation should check:
* `customer_id`、`recommended_action`、`eligible`、`action_name` 是否與規則判斷層完全一致（LLM 不得自行變更）。
* 沒有檢索產品時（A1／A5／A6），不得推薦產品。
* `products` 裡的每個 `product_id` 是否都存在於本次檢索到的 Product Knowledge。
* `product_name` 是否與 Product Knowledge 完全相同，不得竄改。
* 產品 `risk_level` 不得高於客戶 `risk_level`
* `sources` 只能引用本次檢索到的產品。

#### Grounding Policy
* 查核失敗處理方式：
  * 生成 → 查核。
  * 未通過 → 把「具體錯誤清單」附在 prompt 後重試一次 → 再查核。
  * 仍未通過 → 拒絕 LLM 寫的說明，改顯示固定訊息「AI 行動建議說明未通過事實查核，僅提供規則判斷結果，請理專自行判斷。」；規則判斷層的結果（建議行動、成立條件）仍會保留並顯示。
  * 每一次嘗試（不論成敗）都追加一行到 `outputs/nba_audit_log.jsonl` 供人工稽核。
* 程式碼：`rag/nba_generator.py` 的 `generate_grounded_next_best_action()`，結果以 `NBAOutcome`回傳（`status`：`grounded`／`grounded_after_retry`／`rejected`）。

Pydantic model:
```
class RecommendationCheckResult(BaseModel):
    customer_id: str
    is_grounded: bool
    issues: list[str]
```

#### Evaluation
程式碼：`evaluation/what_evaluation.py`。

| 指標 | 定義 | 實測 21 位客戶|
|---|---|---|
| Groundedness | 通過 Recommendation Validation 的客戶比例 | raw 21/21（100%）、Grounding Policy 21/21（100%）|
| Business Rule Compliance | 行動與規則判斷層一致(行動確實成立且優先順序最高、產品類別正確且風險不高於客戶) | 21/21（100%） |
| Product Retrieval Correctness | LLM 挑的產品都在檢索結果內的比例 | 36/36（100%） |
| Product Feature Accuracy | LLM 輸出的產品資訊與 KB 一致的比例 | 36/36（100%） |
| Latency | 資料查詢、產品檢索、LLM 生成、端到端 | 平均 0.009s／0.472s／1.592s／1.869s |

輸出：`outputs/what_evaluation_report.csv`（逐客戶）、`outputs/what_evaluation_summary.json`、`outputs/what_retrieval_report.csv`、`outputs/what_retrieval_summary.json`。

#### Experiment
RM 根據經驗自行決定聯繫策略 vs Customer Evidence + NBA
提升 RM 決策效率與行動一致性，且推薦內容具有資料與產品知識依據。

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
`get_customer_profile` 由下列工具在內部呼叫，不直接開放給 Agent。

Agent 可使用的工具共三個，每個都是把已完成、已通過驗證的流程包成一個呼叫。
程式碼：`agent/agent_tools.py`。
* 取得今日優先名單 `get_top_priority_customers()`
  * Purpose：列出今天最值得優先聯繫的客戶（WHO）。
  * Input：無。
  * Output：每位客戶的 `priority_rank`、`customer_id`、`priority_score` 與三個子分數（`asset_potential_score`、`investment_gap_score`、`investment_intent_score`）。不呼叫 LLM。
* 取得客戶洞察 `get_customer_insight(customer_id)`
  * Purpose：說明這位客戶為什麼值得聯繫（WHY），包含事實查核與 Grounding Policy。
  * Input：customer_id: string（1,000 位客戶中任何一位）
  * Output：Customer Insight + Relevant Fields；若查核未通過則回傳固定訊息。
* 取得行動建議 `get_next_best_action(customer_id)`
  * Purpose：建議下一步行動與候選產品（WHAT）。
  * Input：customer_id: string
  * Output：Next Best Action（含規則判斷結果、候選產品與說明）。若本次對話中尚未取得該客戶的洞察，會先執行 WHY 再執行 WHAT；若 WHY 未通過，WHAT 仍會執行，並註明沒有使用客戶洞察。

## 七、RAG: Product Information
當 AI 要回答「這個產品是什麼、適合什麼樣的客戶、有哪些注意事項」時，知識從哪裡來？
### Knowledge Base
本 POC 的 Product Knowledge 由結構化產品資料與產品說明文件組成。
```
Product Knowledge
├── Fund
│   ├── Fund A
│   ├── Fund B
│   └── Fund C
├── TD
│   └── TD A
├── FX
│   └── FX A
└── FAQ
```
上圖為概念示意。實際為 30 檔虛構產品（`rag/product_knowledge.py`，非真實產品資料），Fund、TD、FX 各 10 檔，風險與幣別分布：

| 類別 | Low | Medium | High | 幣別 |
|---|---|---|---|---|
| Fund | 2 | 4 | 4 | TWD、USD |
| TD | 9 | 1 | 0 | TWD、USD、CNY |
| FX | 4 | 4 | 2 | USD、JPY、EUR、AUD、CNY |

因套用「產品風險不得高於客戶風險」，Low 風險客戶在 Fund 類只有 2 檔可選。
### Schema
Metadata 可用於 Vector Search / Filtering / Reranking。Content 負責「找到之後告訴 LLM 什麼」。
```
Product Knowledge Schema
│
├── Metadata
│   ├── product_id 產品唯一識別碼
│   ├── product_category 基金Fund/定存TD/外幣FX
│   ├── risk_level 產品風險屬性
│   ├── currency TWD/USD/EUR等
│   ├── product_status active/inactive
│   ├── effective_date 資訊生效日期
│   ├── version 資料版本
│   └── source 資料來源
│
└── Content
    ├── product_name 產品名稱
    ├── product_features 產品特色與主要投資內容
    ├── target_customer 適用客群描述
    ├── investment_horizon 建議／適用投資期間
    └── sales_cautions 銷售注意事項
```
### RAG Input
RAG 不直接使用完整 Customer Record 進行產品推薦。

RAG Query 應根據：
* Customer Evidence
* Eligibility / Business Rules

產生產品搜尋條件。例如：

Customer Evidence:
* risk_level = Medium
* investment_ratio = 0.08
* has_fund = 0
* investment_intent = high

Eligibility / Business Rules: A2 基金產品介紹
Search Query:「中風險、適合投資需求、投資期間 3 年以上的基金產品」。

上例為原設計示意。實際的搜尋查詢由程式碼以固定樣板產生，例如 `高風險、適合投資需求的基金產品`，不含投資期間，也不使用洞察文字（見「五、Task／3. Next Best Action／搜尋查詢怎麼產生」）。

### RAG Output
不要只有：「推薦基金 A。」
而是：「根據產品資料，基金 A 的風險屬性為中高風險，主要投資於 XXX，適合投資期間 XXX。由於客戶目前風險屬性為中高風險，因此可作為後續需求討論的候選產品，但仍需進一步確認適合度。」

### 檢索實作
* 切塊：一個產品 = 一個 chunk。每檔產品內容只有數個短欄位，再切只會破壞語意；Metadata 不轉成向量，只用於過濾。
* 向量庫：ChromaDB（本機、免費、無需伺服器），存放於 `data/synthetic/product_knowledge/chroma_db`；索引已建立就直接沿用，不重複呼叫 Embedding API。
* Embedding：`gemini-embedding-001`，可用 `.env` 的 `EMBEDDING_MODEL` 覆寫。
* Top-K：3。
* Filtering：在向量檢索時直接以 Metadata 過濾，條件為產品類別（Fund／TD／FX）與產品風險不高於客戶風險，因此 Top-K 都是合格產品。
* Reranking：未實作，目前只做 Filtering；候選僅 3 檔，由 LLM 從中挑選。

## 八、Agent Workflow
理專用自然語言提問，Agent 判斷要呼叫哪個工具，工具執行既有流程，最後由程式碼把結果整理成回覆。
程式碼：`agent/rm_agent.py`（Agent 流程與對話入口 `main()`）、`agent/agent_llm.py`（LLM 如何選擇工具）、`agent/agent_tools.py`（三個工具與對話狀態、延遲記錄）、`agent/render.py`（回覆由程式碼組成）。

### 完整流程
1. **理專輸入問題**（網頁聊天框`app.py` / 命令列 `rm_agent.py`）。
2. **Agent 把問題交給 LLM 判斷要呼叫哪個工具**（`agent_llm.py`）：
   * Gemini Function Calling（把三個工具宣告給模型，模型回傳要呼叫的工具與參數）。
   * 附上對話狀態（本次對話已討論過哪些客戶、最近幾則提問），讓 LLM 能處理「他」「這位客戶」這類代名詞、以及需要先查名單才知道客戶編號的問法。
3. **LLM 回傳這一輪的判斷**：
   * 不呼叫任何工具，只回一段文字（資訊不足時反問客戶編號，或說明這項請求不支援）。
   * 呼叫一或多個工具（`get_top_priority_customers`／`get_customer_insight`／`get_next_best_action`）。
4. **Guardrails 把關後才真正執行**（`agent_tools.py`）：
   * 只能呼叫上述三個工具。
   * 需要客戶編號的工具，`customer_id` 必須存在於 1,000 位客戶資料中。
   * LLM/API 遇到暫時性錯誤會重試，仍失敗則回報錯誤但不中斷對話。
   * 單次提問的工具呼叫次數有上限。
   * 同一個工具 + 客戶編號重複呼叫會被略過。
5. **各工具執行既有流程**：
   * `get_top_priority_customers`：讀已算好的 WHO 名單，不呼叫 LLM。
   * `get_customer_insight`：跑 WHY（LLM 生成洞察 → Fact Validation → 必要時重試）；本次對話已經問過且通過查核的客戶不會重跑，直接沿用。
   * `get_next_best_action`：若本次對話還沒有這位客戶的洞察，先自動跑一次 WHY，再跑 WHAT（規則選行動 → 視行動觸發 RAG → LLM 生成說明與挑產品 → Recommendation Validation → 必要時重試）；WHY 被拒絕時 WHAT 仍會執行，只是沒有洞察可引用。
   * 每個工具執行成功後，都會把「取得客戶資料、WHY／WHAT 的 LLM 生成、RAG 檢索、端到端」各階段耗時記錄到 `outputs/agent_latency_log.jsonl`（`agent/latency.py`）。
6. **所有工具都執行完後，由程式碼組成最終回覆**（`render.py`）：
   * 把每個工具的結果依呼叫順序轉成理專看到的文字——查核狀態、成立條件、候選產品、固定免責聲明等。
   * LLM 不會插手文字內容，無法改寫查核過的內容。
7. **理專看到回覆，自行判斷是否聯繫、是否採納建議。**

### 實際範例
理專問：「C000104 下一步該怎麼做？」。Agent 判斷只需要呼叫 `get_next_best_action`，但本次對話還沒問過這位客戶，所以自動先跑一次 WHY 當背景資料：
```
【行動建議】C000104　建議行動：A2 基金產品介紹
成立條件：
  - has_fund=0
  - investment_intent_score=92.0 >= 70
  - risk_level='High' in ['High', 'Medium']
說明：客戶目前尚未持有任何基金產品（has_fund=0），且投資意向分數高達92.0（高於70分
的門檻）。客戶風險屬性為高（High），其總資產達20,000,000元...
候選產品（來自產品知識庫，仍需進一步確認適合度）：
  1. 全球科技股票基金（FUND_008｜Fund｜風險 High｜USD）
     此產品主要投資於全球科技巨頭與成長型企業，產品風險屬性為 High，與客戶的高（High）
     風險承受度相符。鑑於客戶本身為科技業主管，此產品可作為候選，仍需理專進一步確認適合度。
  2. 中小型成長股票基金（FUND_010｜Fund｜風險 High｜TWD）
     此產品旨在投資具高成長潛力之中小型企業股票，產品風險屬性為 High，與客戶的高（High）
     風險承受度相符。本產品可作為其優化資產配置之候選，仍需理專進一步確認適合度。
本次已先產生客戶洞察，作為說明的參考。

以上為決策輔助資訊，產品僅為候選，最終適合度與是否聯繫請理專自行判斷。
```
逐段對照回上面的步驟與來源：

| 文字 | 步驟／來源 |
|---|---|
| `建議行動：A2 基金產品介紹`、`成立條件`  | 業務規則層 `business_rules.py`+`action_selector.py`**(非 LLM)** | 
| `說明：客戶目前尚未持有...` | **WHAT：LLM**，prompt 除了成立條件，也附上 WHY 洞察內容供參考，已通過 Recommendation Validation |
| `全球科技股票基金（FUND_008` 的產品/理由 | **RAG 檢索候選產品 + LLM 從中挑選並撰寫理由** |
| `｜Fund｜風險 High｜USD）` | 程式碼查產品目錄 `product_knowledge.py`**(非 LLM)**|
| `本次已先產生客戶洞察，作為說明的參考。` | 代表這次執行內部已先呼叫 WHY `get_customer_insight`，洞察通過 Fact Validation 後，才當作背景資料放進 WHAT： LLM prompt |
| 最後一段「以上為決策輔助資訊...」 | `render_response()` 統一加上免責聲明 |

## 九、使用者介面
理專用網頁與 Agent 對話，同時看到今日優先名單。
程式碼：`ui/app.py`（Streamlit 網頁）、`ui/view_models.py`（把工具結果轉成要顯示的區塊）。
```
┌ 側邊欄 ─────────┬ 主畫面 ───────────────────────────────┐
│ 今日優先名單      │ 你：C000104 下一步怎麼做？           │
│ 1 C000104  90.5  │ 助理：行動建議 A2 基金產品介紹        │
│ 2 C000518  90.5  │   成立條件 / 說明 / 候選產品         │
│ （點一下客戶帶入   │   ✔ 通過查核   ▸ Agent 做了什麼     │
│  常用問題）       │ [輸入框]                            │
└──────────────────┴────────────────────────────────────┘
```
* 名單由程式碼直接顯示，不呼叫 LLM；點選客戶後可一鍵詢問「為什麼值得聯繫？」「下一步該怎麼做？」，也可自由輸入。
* 結構化結果呈現：
  * 名單用表格。
  * 洞察與行動建議顯示查核狀態（通過/修正後通過/未通過）。
  * 可展開查看欄位依據、成立條件、候選產品。
  * 未通過查核時不顯示 AI 寫的內容，只顯示規則判斷結果。
* Agent 處理時即時顯示工具呼叫過程，完成後可展開回看。
* 畫面固定提醒：產品僅為候選，最終適合度與是否聯繫由理專判斷。
* 不做：登入、多人同時使用、公開雲端部署。

## 十、系統效能
記錄 Agent 每次成功回答問題時各階段耗時，與理專原本作業時間比較，以及日後追蹤優化效果。
程式碼：`agent/latency.py`。
輸出：`outputs/agent_latency_log.jsonl`。

### 各階段實測
（單次真實呼叫，僅供參考，會隨網路狀況變動）
| 階段 | 實作方式 | 實測耗時 |
|---|---|---|
| 取得客戶資料 | 讀 `priority_customers_all.csv` | < 0.05s |
| WHY／WHAT LLM 生成 | Gemini API，一次網路往返；查核失敗會重試一次 | 約 1.7～2.7s/次 |
| RAG 檢索 | 一次 Embedding API 呼叫 + 本機 Chroma 向量比對 | 約 0.3～0.5s |
| RAG 檢索（冷啟動） | 建立 Embedding client + 開啟本機索引，只在一個伺服器行程中發生一次 | 約 0.7～1.4s |

### 優化方向
* **查詢快取：** 搜尋查詢由固定樣板產生（風險等級 × 行動類別 × 少數幣別提示），組合數很少且可窮舉，同組合一定得到同向量，可快取掉大部分 Embedding API 呼叫。
* **提前暖機：** 伺服器啟動時就建立 Embedding client、載入 Chroma，不要等第一次提問才做。
* **模型可換：** Embedding 模型與 LLM 模型一樣可在 `.env` 設定（`EMBEDDING_MODEL`／`GEMINI_MODEL`／`ANTHROPIC_MODEL`），換模型比較延遲與品質不需改程式碼，對應 Phase 6（實驗與優化）的模型實驗。

### 潛在瓶頸
資料量變大時，30 檔產品尚未遇到，屬於未來規劃。
* **索引建置成本：** 每檔產品都要呼叫一次 Embedding API；產品從幾十檔變幾千、幾萬檔，建索引時間與 API 配額會先出問題，而不是向量比對本身。優化方向：批次／平行呼叫 Embedding API、只對新增或異動的產品重新嵌入（增量索引），不要整批重建。
* **向量比對變慢：** 目前是精確比對（brute-force 等級），30 檔幾乎瞬間；量大之後才需要調整近似索引（如 HNSW）的參數，在準確度與速度間取捨。
* **儲存與記憶體：** 3072 維浮點向量 × 筆數，量大時磁碟與記憶體占用會上升；優化方向：改用維度較低的 Embedding 模型，用儲存空間換取一些語意精度。
* **單機嵌入式向量庫的上限：** ChromaDB 目前是本機單行程模式，若要多台伺服器共用同一份索引或大幅擴充，需改用有伺服器模式的向量庫或代管服務，屬於超出本次 POC 範圍的規劃方向。
