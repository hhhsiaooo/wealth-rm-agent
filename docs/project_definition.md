# 財富管理客戶經營決策支援
### 一、Business Context
財富管理部理專經營大量信託客戶(15 位理專負責約 2,000 位信託客戶)，執行 Outbound Call 時需要切換不同系統查詢客戶資料(CRM、財管系統)，以掌握資產、產品持有、資金流、投資行為及數位軌跡等資訊，**判斷這個客戶現在是否值得聯繫、有什麼商品機會。**

### 二、Original Problem
**資訊分散於 CRM、財管系統及其他資料來源**，理專需要跨系統查詢並人工整合客戶資訊。

### 三、Original Solution
**整合資料源並建立 Customer 360 Dashboard**，集中提供客戶基本資料、產品持有、資產結構、消費、資金流、投資及數位行為等資訊。

### 四、Outcome
Dashboard 已降低理專資訊查找及理解成本：
* **Customer Profile**：年齡、職業、KYC、地區 -> 了解客戶
* **Product Portfolio**：定存、基金、保險、貸款等 -> 找產品缺口
* **Asset**：AUM、台外幣活定存、理財庫存 -> 判斷資產規模與配置
* **Behavior**：消費、資金流、代扣繳 -> 推測客戶需求與黏著度
* **Investment Intent**：基金申購、數位軌跡 -> 判斷近期投資意圖
```
15 位理專皆有使用，平均每人每月使用約 5～10 次。問卷顯示：
14 位認為有助於產生客戶洞察報告
13 位認為有助於快速理解陌生客戶
11 位認為有助於業務媒合與銷售轉換。
```
### 五、Problem Discovery
* **Problem A：Dashboard 資訊太多且 Insight 仍需要人工整合**
  * Dashboard 已有效改善「資訊分散」的問題。
  * 理專仍需自行整合多維度客戶資訊，仰賴個人經驗判斷「誰值得優先聯繫」、「要推薦什麼產品」。
* **Problem B：Dashboard 的推薦邏輯偏規則式，且未驗證有效性**
  * 推薦邏輯偏規則式。
    * AUM 高 + 大量台幣活存 + 沒有基金 + 中高風險屬性 = 理專可以進一步了解投資需求
    * 基金瀏覽 + 申購未完成 = 可能對該產品有興趣
  * 目前只是業務假設，未驗證推薦邏輯的實際成功率。
    * 基金瀏覽 + 申購未完成 -> 實際聯繫 -> 多少最後真的申購？
  * 未控制其他影響因素，無法將單月新增的 55 位客戶直接歸因於 Dashboard 成效。

### 六、Business Opportunity
**降低理專從 Customer Insight 到 Sales Action 的人工判斷成本。**
* 業務流程：理專 -> 多個系統查詢 -> 人工整理資訊 -> 人工判斷客戶 -> 打電話
* Dashboard：理專 -> Dashboard 整合資訊 -> 人工判斷客戶 -> 打電話
* **Opportunity：Decision Making**
  * **Who	今天應該優先聯繫誰？**
  * **Why	為什麼這個客戶值得聯繫？**
  * **What	聯繫時可以採取什麼行動？**
```
                Existing Dashboard
                       ↓
               Customer Information
                       ↓
                 Customer Insight
                       ↓
                     [Gap]
                       ↓
                  Sales Action
```
