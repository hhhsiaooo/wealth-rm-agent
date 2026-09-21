"""Phase 3 (WHAT): shared Pydantic schemas.

Two layers, kept as separate models on purpose (same principle as WHY's
CustomerInsight / FactCheckResult split -- see docs/ai_product_design.md,
"五、Task／2. Customer Evidence／Schema 設計原則"):

- Rule layer (deterministic, no LLM): ActionEligibility / ActionSelectionResult.
  Produced entirely by business_rules.py + action_selector.py.
- LLM layer (Phase 3.8, not implemented yet): NextBestAction /
  ProductRecommendation. recommended_action/eligible on NextBestAction must
  be copied verbatim from ActionSelectionResult, never re-decided by the LLM.
- Validation layer (Phase 3.9, not implemented yet): RecommendationCheckResult,
  mirrors insight/fact_validator.py's FactCheckResult.
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


class RecommendationCheckResult(BaseModel):
    customer_id: str
    is_grounded: bool
    issues: list[str]
