# WHY / WHAT 評估指標優化

> 進行評估指標優化，補齊「LLM 自由文字」與「RAG 檢索品質」這兩個尚未被系統性查核的構面。

---

## 一、第一版評估指標

本專案的評估分兩層：
- **查核規則（Validation）**：
  - 建立規則式查核程式碼，檢查 LLM 輸出的結構化欄位清單，是否與原始資料一致。
  - `insight/fact_validator.py`、`rag/recommendation_validator.py`。
- **評估彙總（Evaluation）**：
  - 實際呼叫 LLM 生成要被查核的客戶洞察與行動建議，進行規則式查核，產出評估報告。 
  - `evaluation/why_evaluation.py`、`evaluation/what_evaluation.py`。

### 1.1 WHY Evaluation 客戶洞察評估指標
- 檢查對象：LLM 輸出的 `CustomerInsight` 結構化欄位（`customer_id`、`priority_rank`、`priority_score`、`key_fields`）。

> **現況缺口：**`CustomerInsight.insight`（給理專看的自由文字敘述）完全沒有被查核過。 只有拿去組成 insight 的「材料」（key_fields）被查核，材料查完之後 LLM 怎麼把它寫成一句話，沒有任何機制把關。

- **查核規則（Validation）**
程式碼：`insight/fact_validator.validate_customer_insight()`

| 指標 | 定義 |
|---|---|
| customer_id_mismatch | 回傳的 `customer_id` 與查詢對象不符 |
| unknown_field | key_fields 裡的欄位名稱，在原始 43 欄資料中根本不存在（幻覺） |
| value_mismatch | 欄位存在，但值與原始資料不符 |
| priority_score_mismatch、priority_rank_mismatch | 分數和排名與原始資料不符 |

- **評估指標（Evaluation）**
程式碼：`evaluation/why_evaluation.compute_summary()`

| 指標 | 定義 |
|---|---|
| `Factual Accuracy` | 輸出通過查核的比例(raw原始輸出/final重試修正) |
| `Hallucination Rate` | 首次輸出的 key_fields 中，unknown_field 佔比 |
| `Value Mismatch Rate` | 首次輸出的 key_fields 中， value_mismatch、priority_score_mismatch、priority_rank_mismatch 佔比 |
| `Retry Rate`、`Rejection Rate` | 需要重試 / 最終被拒絕顯示的客戶比例 |
| `Latency` | 量測查詢客戶資料、端到端的耗時 |


### 1.2 WHAT 行動建議評估指標
- 檢查對象：LLM 輸出的結構化欄位 `NextBestAction.products`。

> **現況缺口：**`reason`（LLM 依據產品資料說明可作為候選的原因）/ `explanation`（LLM 引用行動成立條件與客戶洞察，給理專看的自由文字敘述）完全沒有被查核過。只有檢查 LLM 挑選的 products 是否來自 RAG 檢索結果，並與商品原始資料一致。

- **查核規則（Validation）**
程式碼：`rag/recommendation_validator.validate_next_best_action()`

| 指標 | 定義 |
|---|---|
| 商品未被竄改 | products 裡每個 product_id 都在本次檢索結果內、product_name 與原始資料相同 |
| 風險上限 | products 的 risk_level 不得高於客戶 risk_level |
| 無檢索時不得有商品 | A1投資需求訪談/A5既有投資組合檢視/A6暫不聯繫 未觸發 RAG 檢索時，products 必須為空陣列 |

- **評估指標（Evaluation）**
程式碼：`evaluation/what_evaluation.py`

| 指標 | 定義 |
|---|---|
| `Groundedness` | 輸出通過查核的比例(raw原始輸出/final重試修正)(包含 recommended_action／eligible／action_name 等防禦性斷言) |
| `Product Selection In Retrieved` | LLM 挑的商品是否都在本次檢索結果內 |
| `Product Name Accuracy` | 挑選商品的名稱是否與原始資料一致 |
| `Product Risk Accuracy` | 挑選商品的風險是否符合客戶風險屬性 |
| `Retry Rate`、`Rejection Rate` | 需要重試 / 最終被拒絕顯示的客戶比例 |
| `Latency` | 量測查詢客戶資料、產品檢索、LLM 生成、端到端的耗時 |

---

## 二、新增評估指標
- 基於現況缺口新增指標，不修改、不取代任何一個現有的規則查核或既有評估欄位。
- 方法：RAGAS、LLM-as-judge。

| 現況缺口 | 新增指標 | 方法 |
|---|---|---|
| WHY insight | 客戶洞察的語意論述是否有憑有據 | `RAGAS Faithfulness` |
| WHAT explanation / reason | 行動建議的語意論述是否有憑有據| `RAGAS Faithfulness` |
| 文字撰寫品質 | LLM 撰寫的文字除了要有憑有據，是否讓使用者容易理解 | `LLM-as-judge rubric` |
| RAG 檢索品質 | 檢索回來的商品是否都適合、有沒有漏掉更適合的 (人工標記 golden dataset 比對 RAG 實際檢索到的 product_id) | `Golden Dataset` + `Context Precision/Recall`|

---

## 三、新增指標細節

### 3.1 RAGAS Faithfulness
RAGAS（Retrieval-Augmented Generation Assessment）是專門評估 LLM/RAG 系統輸出品質的套件。核心想法：很多「這段話好不好」的判斷，規則式判斷不出來，因為同一件事可以用完全不同的措辭表達。RAGAS 的解法是用另一個 LLM 當裁判，把原本主觀、沒辦法人工規模化的判斷，變成可重複執行、可量化的分數。
* 把要查核的文字（response）拆成一條條獨立的「原子主張」（atomic claims）
* 對每條主張，問裁判 LLM：「這句話能不能從給定的 context 推得出來？」
* Faithfulness = 能推得出來的主張數 ÷ 總主張數
例如：「客戶資產約800萬元，且是金融業高階主管，非常適合高風險商品」被拆成約 3 條主張，只有「資產約800萬」這條有 context 撐得住，另外兩條是編的，1/3 = 0.333。

#### 評估對象
優先名單 12 位 + 補上 3 位 (每種行動最多補 1 位)，本次實際為 15 位。
每段文字的評估依據 context：
* insight → 客戶 43 欄資料
* explanation → 成立條件 ＋ WHY 洞察
* reason → 商品資料 ＋ 客戶風險等級

#### 評估流程
* RAGAS 封裝層
  * 載入 ragas 套件 
  * 建立裁判 LLM
  * 建立函式 score_faithfulness()，把 response（要判斷的內容）、contexts（判斷依據）、llm（判斷者）、user_input（這段內容在回答的問題） 丟給 ragas，回傳分數
  * 程式碼：`ragas_faithfulness.py`

| ragas 功能 | 函式 | 用途 |
|---|---|---|
| `LangchainLLMWrapper` | `ragas_faithfulness.get_faithfulness_llm()` | 把 Gemini 的模型包成 ragas 能用的裁判介面 |
| `Faithfulness` | `ragas_faithfulness.score_faithfulness()` | 指標本體，內含兩個 prompt（拆 statements、逐條判定 0/1）與計分公式 |
| `SingleTurnSample` | `ragas_faithfulness.score_faithfulness()` | ragas 的標準輸入格式，把 user_input、response、contexts 包在一起 |
| `metric.single_turn_ascore(sample)` | `ragas_faithfulness.score_faithfulness()` | 執行整個流程：先用 LLM 拆成 statements，再用 LLM 逐條判斷，最後算「判 1 的數量 ÷ 總數」 |

* 執行評估
  * 選客戶、執行 WHY/WHAT 流程生成文字 responses
  * 把每段 response 和它專屬的 context、user_input 配對好
  * 呼叫 score_faithfulness 取得分數（透過 _score_with_retry 處理失敗）
  * 彙總報告 summary、checkpoint 續跑
  * 程式碼：`ragas_evaluation.py`

* 修正過程
  * reason Faithfulness 偏低可能原因：reason 的 prompt 要求模型「如實陳述產品風險屬性與客戶風險屬性的關係」，所以理由常會出現「與客戶的高風險承受度相符」這類句子。原 Faithfulness 評估用的 context 只放入商品自己的資料，客戶風險等級不在內，自然會判為無依據。
  * 修正評估設計的缺陷：RAGAS 驗證用的 context 加入客戶風險等級。
  * 修正評估設計的缺陷：Faithfulness 補上參數 user_input（四個參數必填 response, contexts, llm, user_input）。

#### 評估結果
完整評估：`poetry run python src/evaluation/ragas_evaluation.py --per-action 1 --judge-model gemini-3.5-flash-lite`

沿用生成文字，僅換裁判模型：`poetry run python src/evaluation/ragas_evaluation.py --rejudge-from outputs/ragas_evaluation_report.csv --judge-model <新模型>`

檔案：`outputs/ragas_evaluation_report.csv`
- 每一筆紀錄是一段被評分的文字（不是每個客戶一列，一個客戶可能 1 筆 insight + 1 筆 explanation + 0~3 筆 reason）。
- 欄位：stage、text_type、customer_id、recommended_action、product_id、faithfulness、response_text、contexts_json

檔案：`outputs/ragas_evaluation_summary.json`
- 按照 insight/explanation/reason 三種文字類型分開算平均 Faithfulness 分數。

```
# 修正前：reason 的 context 不含客戶風險等級、user_input 為空字串
{
  "total_customers": 15,
  "total_scored_texts": 63,
  "judge_failures": 0,
  "mean_faithfulness_overall": 0.6487852975948214,
  "mean_faithfulness_why_insight": 0.8948412698412699,
  "mean_faithfulness_what_explanation": 0.6729285529285529,
  "mean_faithfulness_what_reason": 0.5259674668765577,
  "generation_model": "gemini-3.5-flash-lite",
  "judge_model": "gemini-3.5-flash-lite"
}

# 修正後：reason 的 context 含客戶風險等級、user_input 為對應問題
{
  "total_customers": 15,
  "total_scored_texts": 65,
  "judge_failures": 0,
  "mean_faithfulness_overall": 0.7555021047328739,
  "mean_faithfulness_why_insight": 0.8768518518518519,
  "mean_faithfulness_what_explanation": 0.792069967069967,
  "mean_faithfulness_what_reason": 0.6878231292517006,
  "generation_model": "gemini-3.5-flash-lite",
  "judge_model": "gemini-3.5-flash-lite"
}
```

#### 限制
- gemini-3.5-flash-lite 的 temperature 無法控制，同一份文字重跑時 Faithfulness 分數會些微波動，比較時不要把些微差異當成真的進步。
- 裁判與生成模型相同，模型可能傾向認可自己寫出來的句型，高估自己的忠實度，未來可用不同供應商或更強的模型當裁判，或人工抽查文字，逐條人工標記主張是否有依據，再與 LLM 的判斷比對。

### 3.2 LLM-as-judge rubric
針對 `insight/insight_generator.py` 產出的 insight，以及 WHAT 的 `explanation`，設計簡單 rubric（例如 clarity、actionable，1–5 分），讓 LLM 扮演評審角色評分。

### 3.3 Golden Dataset + Context Precision/Recall
- 現有的 `product_retrieval_correctness` 只測「檢索器有沒有照著 metadata 過濾規則做事」（category、risk 對不對）。
- Context Precision / Recall 測的是「檢索回來的東西，是不是真的跟這個客戶最相關」，以及「有沒有漏掉真正該出現的商品」——這需要人工標記一批 (客戶條件 → 應該檢索到哪些商品) 的 ground truth，也就是這一步要建的 golden dataset。
- 抽樣規劃：涵蓋 A2／A3／A4 三種會觸發檢索的行動、三種風險等級，與現有`what_evaluation.py`的`_retrieval_cases()`案例矩陣對齊，方便重用既有的檢索評估流程。

---

## 四、評估指標總覽 Roadmap

| 構面 | 指標 | 查核欄位 | 狀態 | 
|---|---|---|---|
| WHY 結構化查核 | `Factual Accuracy`、`Hallucination Rate`、`Value Mismatch Rate`、`Retry Rate`、`Rejection Rate` | customer_id、unknown_field、value_mismatch、priority_score、priority_rank | 已完成 |
| WHY 自由文字 | `RAGAS Faithfulness` | insight | **本次新增，已完成** |
| WHY 文字品質 | `LLM-as-judge rubric` | insight | **本次新增，未實作** |
| WHAT 結構化查核 | `Groundedness`、`Product Selection In Retrieved`、`Product Name Accuracy`、`Product Risk Accuracy`、`Retry Rate`、`Rejection Rate` | product_id、product_name、risk_level | 已完成 |
| WHAT 自由文字| `RAGAS Faithfulness` | explanation、reason | **本次新增，已完成** |
| WHAT 文字品質 | `LLM-as-judge rubric` | explanation、reason | **本次新增，未實作*** |
| RAG 檢索過濾正確性 | `Product Retrieval Correctness` | product_category、risk_level、currency | 已完成 |
| RAG 檢索相關性 | `Golden dataset + Context Precision/Recall` |product_id | **本次新增，未實作*** |
| 各階段耗時 | `Latency` | 查詢客戶資料、產品檢索、LLM 生成、端到端 | 已完成 |

---

## 五、實作注意事項

1. **不動既有指標**：新增指標一律以獨立函式/欄位加入，`fact_validator.py`、`recommendation_validator.py`、既有的 `WhyEvaluationSummary`／`WhatEvaluationSummary` 欄位不修改。
2. **pytest 一律 stub，不打真的 API**：延用專案既有模式（`tests use a stub LLM`），RAGAS 的 Faithfulness／Context Precision 內部也會呼叫 LLM 當裁判，測試時要 mock 掉，不能因為導入 RAGAS 就破例。
3. **Gemini 額度節流**：`why_evaluation.py`／`what_evaluation.py` 已有 `delay = 13.0 if isinstance(llm_client, GeminiLLMClient) else 0.0` 的節流機制。RAGAS 的 LLM 裁判呼叫要沿用同一套節流邏輯，避免新增指標讓原本就吃緊的每日額度（目前已知 WHAT pipeline 曾卡在 20 req/day）更快見底。
4. **Golden dataset 與案例矩陣對齊**：Context Precision/Recall 的標記案例，盡量對齊 `what_evaluation._retrieval_cases()` 既有的 (category × risk_level) 矩陣，之後才能跟既有檢索評估報告放在一起比較。

---

## 六、待辦順序（Phase A）

- [ ] 1. 看 Hahow 課程第七章（Benchmark、RAGAS、LLM-as-judge）
- [v] 2. 加 RAGAS Faithfulness：WHY 的 `insight` ＋ WHAT 的 `explanation`/`reason`
- [ ] 3. 設計 LLM-as-judge rubric（clarity、actionable），涵蓋 WHY 與 WHAT 的文字品質
- [ ] 4. 建 RAG 檢索 golden dataset（客戶條件 → 應檢索到的商品）
- [ ] 5. 加 RAGAS Context Precision / Recall
- [v] 6. 新增 `evaluation/ragas_evaluation.py`，串 `rag/retrieval.py`、`rag/nba_generator.py` 的輸出
- [v] 7. 補 pytest（RAGAS 呼叫全部 stub）
- [v] 8. 跑一次完整 baseline（既有 why/what evaluation ＋ 新增 RAGAS／LLM-judge 指標），存成快照
- [ ] 9. Commit，作為 Phase B（RAG 優化）、Phase C（LangGraph 重構）、Phase D（Finetuning）的比較基準

---

## 七、這份 Baseline 的用途

Phase A 結束時存下的快照，是後續所有優化的「改動前」對照組：

- **Phase B**（商品資料庫擴充、chunk 策略、Re-Ranking）：改動後重跑本文件第四節的全部指標，尤其是 Context Precision/Recall 與 product_retrieval_correctness，比對是否真的變好。
- **Phase C**（LangGraph 重構 Agent）：重點看行為一致性（groundedness、business_rule_compliance 不能退步）與延遲，不必然要求分數變好。
- **Phase D**（Finetuning 取代 Gemini）：直接拿 Gemini baseline 的全部指標，對照微調模型的同一份指標，包含新增的 Faithfulness／LLM-as-judge 分數。

沒有這份 baseline，後面每一次「感覺變好了」都只是主觀印象，無法量化證明。
