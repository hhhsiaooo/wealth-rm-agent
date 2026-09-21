"""Phase 3.6/3.7 (WHAT): Synthetic Product Knowledge dataset + chunking.

30 synthetic products (10 each of Fund/TD/FX), matching the Metadata/Content
schema in docs/ai_product_design.md "七、RAG: Product Information". This is
fabricated POC content, not real product data.

Chunking strategy (3.7): one product = one chunk. Each product's Content
fields are short (a handful of fields), well under any reasonable chunk size,
so splitting further would only fragment the meaning. `to_chunk_text()"
builds the single passage that gets embedded; Metadata fields are stored as
structured metadata for filtering (e.g. by product_category), never embedded
as text.
"""

from __future__ import annotations

from pydantic import BaseModel

DEFAULT_STATUS = "active"
DEFAULT_EFFECTIVE_DATE = "2026-01-01"
DEFAULT_VERSION = "1.0"
DEFAULT_SOURCE = "synthetic_poc"


class ProductMetadata(BaseModel):
    product_id: str
    product_category: str  # "Fund" | "TD" | "FX"
    risk_level: str        # "Low" | "Medium" | "High"
    currency: str
    product_status: str
    effective_date: str
    version: str
    source: str


class ProductContent(BaseModel):
    product_name: str
    product_features: str
    target_customer: str
    investment_horizon: str
    sales_cautions: str


class ProductKnowledge(BaseModel):
    metadata: ProductMetadata
    content: ProductContent

    def to_chunk_text(self) -> str:
        """3.7: one product = one chunk. Concatenate Content into a single
        natural-language passage for embedding."""
        c = self.content
        return (
            f"{c.product_name}。{c.product_features}"
            f"適合{c.target_customer}，建議投資期間{c.investment_horizon}。"
            f"注意事項：{c.sales_cautions}"
        )

    def to_flat_metadata(self) -> dict[str, str]:
        """Chroma metadata values must be flat str/int/float/bool -- this is
        the ProductMetadata fields as a flat dict for collection.add()."""
        return self.metadata.model_dump()


def _make_product(
    product_id: str,
    product_category: str,
    risk_level: str,
    currency: str,
    product_name: str,
    product_features: str,
    target_customer: str,
    investment_horizon: str,
    sales_cautions: str,
) -> ProductKnowledge:
    return ProductKnowledge(
        metadata=ProductMetadata(
            product_id=product_id,
            product_category=product_category,
            risk_level=risk_level,
            currency=currency,
            product_status=DEFAULT_STATUS,
            effective_date=DEFAULT_EFFECTIVE_DATE,
            version=DEFAULT_VERSION,
            source=DEFAULT_SOURCE,
        ),
        content=ProductContent(
            product_name=product_name,
            product_features=product_features,
            target_customer=target_customer,
            investment_horizon=investment_horizon,
            sales_cautions=sales_cautions,
        ),
    )


# -- Fund (10) : maps to A2 --------------------------------------------------
_FUND_PRODUCTS = [
    _make_product("FUND_001", "Fund", "Low", "TWD", "貨幣市場基金", "投資短期貨幣市場工具，波動低", "保守型、追求流動性與穩定收益的客戶", "不限期間，隨時可贖回", "收益率隨市場利率波動，非保本商品"),
    _make_product("FUND_002", "Fund", "Low", "TWD", "台幣債券型基金", "主要投資投資等級公司債與公債，追求穩定配息", "保守穩健、重視資產保值的客戶", "建議1年以上", "債券價格受利率波動影響"),
    _make_product("FUND_003", "Fund", "Medium", "TWD", "平衡型基金（股債均衡）", "股債比例約5:5動態調整，兼顧成長與防禦", "穩健型、希望股債分散配置的客戶", "建議2年以上", "市場劇烈波動時仍可能虧損"),
    _make_product("FUND_004", "Fund", "Medium", "TWD", "高股息平衡基金", "聚焦高股息股票與債券組合，強調配息穩定性", "追求穩定配息與部分資本增值的客戶", "建議2年以上", "配息可能來自本金，需留意淨值變化"),
    _make_product("FUND_005", "Fund", "Medium", "USD", "全球REITs不動產基金", "投資全球上市不動產投資信託，具一定利率敏感度", "希望分散至不動產資產類別的中風險客戶", "建議3年以上", "受利率與不動產景氣循環影響大"),
    _make_product("FUND_006", "Fund", "Medium", "USD", "新興市場債券基金", "投資新興市場主權及企業債，收益率高於已開發市場", "可承受匯率與信用風險、追求較高收益的客戶", "建議3年以上", "需承擔匯率風險與新興市場信用風險"),
    _make_product("FUND_007", "Fund", "High", "TWD", "台灣科技股票基金", "聚焦半導體、電子等台灣科技供應鏈股票", "積極型、看好科技產業成長的客戶", "建議3年以上", "產業集中度高，波動大"),
    _make_product("FUND_008", "Fund", "High", "USD", "全球科技股票基金", "投資全球科技巨頭與成長型企業", "積極型、追求長期資本增值的客戶", "建議5年以上", "短期波動劇烈，不適合資金需求急迫的客戶"),
    _make_product("FUND_009", "Fund", "High", "USD", "全球ESG永續股票基金", "篩選ESG評等優良企業進行投資", "積極型、重視永續投資理念的客戶", "建議5年以上", "產業配置受ESG篩選限制，可能與大盤表現偏離"),
    _make_product("FUND_010", "Fund", "High", "TWD", "中小型成長股票基金", "投資具高成長潛力之中小型企業股票", "積極型、能承受高波動的客戶", "建議5年以上", "流動性較低、波動度高於大型股基金"),
]

# -- TD (10) : maps to A3 -----------------------------------------------------
_TD_PRODUCTS = [
    _make_product("TD_001", "TD", "Low", "TWD", "台幣活期優利定存", "牌告利率優於一般活存，1個月即可滾動", "短期資金停泊、重視流動性的客戶", "1個月期", "提前解約按活存利率計息"),
    _make_product("TD_002", "TD", "Low", "TWD", "台幣3個月期定存", "固定利率，到期自動展期可選", "短中期資金規劃的客戶", "3個月期", "提前解約需支付違約金"),
    _make_product("TD_003", "TD", "Low", "TWD", "台幣6個月期定存", "利率較3個月期略高", "中期資金規劃的客戶", "6個月期", "中途解約影響利息收益"),
    _make_product("TD_004", "TD", "Low", "TWD", "台幣1年期定存", "利率為台幣定存中相對優惠之區間", "長期資金規劃、追求穩定利息的客戶", "1年期", "資金鎖定期間較長"),
    _make_product("TD_005", "TD", "Low", "TWD", "台幣2年期定存", "長天期定存利率通常較優", "長期閒置資金規劃的客戶", "2年期", "資金流動性低"),
    _make_product("TD_006", "TD", "Low", "USD", "美元3個月期定存", "美元計價，利率通常高於台幣定存", "持有美元、短期停泊的客戶", "3個月期", "到期解約涉及匯率風險"),
    _make_product("TD_007", "TD", "Low", "USD", "美元1年期定存", "鎖定美元利率，適合美元資產配置", "長期持有美元資產的客戶", "1年期", "匯率波動可能影響台幣換算價值"),
    _make_product("TD_008", "TD", "Low", "CNY", "人民幣定存", "人民幣計價定存，分散幣別配置", "有人民幣資產配置需求的客戶", "6個月期", "人民幣匯率波動及流動性限制"),
    _make_product("TD_009", "TD", "Medium", "USD", "外幣組合定存（結合換匯）", "結合換匯操作提高約定利率，但涉及匯率風險", "可承受匯率波動、追求較高約定利率的客戶", "1個月期", "到期匯率不利時可能產生匯兌損失，非保本商品"),
    _make_product("TD_010", "TD", "Low", "TWD", "階梯式定存組合", "資金分批投入不同天期定存，平衡流動性與收益", "希望分散到期時間、兼顧流動性與利率的客戶", "1年期（分批到期）", "部分資金仍需鎖定較長天期"),
]

# -- FX (10) : maps to A4 ------------------------------------------------------
_FX_PRODUCTS = [
    _make_product("FX_001", "FX", "Low", "USD", "外幣活期存款", "隨時存取，適合外幣資金停泊", "有外幣收支需求、追求流動性的客戶", "不限期間", "活存利率低，匯率隨市場波動"),
    _make_product("FX_002", "FX", "Low", "USD", "美元保本結構型商品", "到期保本，依匯率區間表現支付額外收益", "保守型、希望參與匯率行情但重視保本的客戶", "6個月期", "提前贖回可能無法保本"),
    _make_product("FX_003", "FX", "Low", "JPY", "日圓計價定期商品", "日圓計價，可搭配旅遊或留學資金規劃", "有日圓資產需求或看好日圓的客戶", "3個月期", "日圓匯率波動度需留意"),
    _make_product("FX_004", "FX", "Medium", "AUD", "澳幣高息貨幣商品", "澳幣傳統上與台幣存在利差，收益相對優惠", "追求較高利差收益、可承受匯率波動的客戶", "6個月期", "澳幣波動度較主要貨幣高"),
    _make_product("FX_005", "FX", "Low", "EUR", "歐元計價商品", "歐元計價，分散美元集中度", "有歐元資產配置需求的客戶", "3個月期", "歐元區政經情勢影響匯率"),
    _make_product("FX_006", "FX", "High", "USD", "目標可贖回遠期外匯（簡化版TRF）", "依約定匯率區間分批履約，可能優於即期匯率", "具外匯避險或投機需求、風險承受度高之客戶", "依合約條件（通常6-12個月）", "匯率大幅不利走勢時虧損可能放大，屬複雜性高風險商品"),
    _make_product("FX_007", "FX", "High", "USD", "雙元貨幣結構型商品（DCI）", "到期依匯率表現以約定貨幣其中一種返還本金及收益", "可承受本金轉換為連結貨幣、追求較高約定利率的客戶", "1個月期", "到期本金可能被轉換為較弱勢貨幣，非保本商品"),
    _make_product("FX_008", "FX", "Medium", "USD", "外幣保單連結幣別商品", "結合保障與外幣資產累積", "有長期外幣資產規劃需求的客戶", "長期（保單年期）", "涉及保單相關費用與匯率風險，非單純存款商品"),
    _make_product("FX_009", "FX", "Medium", "CNY", "人民幣外匯商品", "人民幣計價，分散美元集中風險", "看好人民幣或有人民幣資金需求的客戶", "3個月期", "人民幣匯率政策及流動性風險需留意"),
    _make_product("FX_010", "FX", "Medium", "USD", "多幣別組合帳戶", "單一帳戶可持有多種外幣，靈活調度", "有多幣別資產配置需求、經常跨幣別操作的客戶", "不限期間", "幣別轉換涉及匯差成本"),
]

PRODUCTS: list[ProductKnowledge] = _FUND_PRODUCTS + _TD_PRODUCTS + _FX_PRODUCTS


def load_products() -> list[ProductKnowledge]:
    return list(PRODUCTS)


def load_products_by_category(product_category: str) -> list[ProductKnowledge]:
    return [p for p in PRODUCTS if p.metadata.product_category == product_category]
