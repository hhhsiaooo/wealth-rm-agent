"""Phase 3 (WHAT): shared Pydantic schemas.

Three layers, kept as separate models on purpose (same principle as WHY's
CustomerInsight / FactCheckResult split -- see docs/ai_product_design.md,
"五、Task／3. Next Best Action／Schema 設計原則"):

- Rule layer (deterministic, no LLM): ActionEligibility / ActionSelectionResult.
  Produced entirely by business_rules.py + action_selector.py.
- LLM layer: NBALLMOutput is the ONLY thing the LLM writes (explanation +
  product picks). nba_generator assembles the final NextBestAction: it copies
  recommended_action / action_name / eligible from ActionSelectionResult and
  derives sources from the picked products, so the LLM cannot re-decide them.
- Validation layer: RecommendationCheckResult, mirrors insight/fact_validator.py's
  FactCheckResult.
"""

from __future__ import annotations

from pydantic import BaseModel


class ActionEligibility(BaseModel):
    action_id: str
    eligible: bool
    matched_conditions: list[str]


class ActionSelectionResult(BaseModel):
    customer_id: str
    evaluated_actions: list[ActionEligibility]
    recommended_action: str


class ProductRecommendation(BaseModel):
    product_id: str
    product_name: str
    reason: str


class NextBestAction(BaseModel):
    customer_id: str
    recommended_action: str
    action_name: str
    eligible: bool
    explanation: str
    products: list[ProductRecommendation] = []
    sources: list[str] = []


class NBALLMOutput(BaseModel):
    """What the LLM is allowed to write for a Next Best Action -- only the
    explanation and the product picks. recommended_action / eligible /
    action_name / sources are NOT LLM-controlled; nba_generator copies the
    first three from ActionSelectionResult and derives sources from the
    picked products. No field defaults: Gemini's response_schema has been
    reported to reject Pydantic defaults."""

    explanation: str
    products: list[ProductRecommendation]


class RecommendationCheckResult(BaseModel):
    customer_id: str
    is_grounded: bool
    issues: list[str]
