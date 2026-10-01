from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# ---------- Step 1: OCR / text extraction ----------
class OCRResult(BaseModel):
    raw_text: str
    raw_tokens: List[str]
    currency_hint: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)


# ---------- Step 2: Normalization ----------
class NormalizedToken(BaseModel):
    raw: str                      # token as Step 1 found it, e.g. "l200"
    value: float                  # parsed number, e.g. 1200.0
    is_percent: bool = False
    corrected: bool = False       # True if we had to fix OCR characters


class NormalizedResult(BaseModel):
    normalized_amounts: List[float]
    percentages: List[float] = []
    tokens: List[NormalizedToken] = []
    normalization_confidence: float = Field(ge=0.0, le=1.0)


# ---------- Step 3: Classification ----------
AmountType = Literal["total_bill", "paid", "due", "discount", "subtotal", "tax", "other"]


class ClassifiedAmount(BaseModel):
    type: AmountType
    value: float
    source: str = ""   # e.g. "Total: INR 1200"


class ClassificationResult(BaseModel):
    amounts: List[ClassifiedAmount]
    confidence: float = Field(ge=0.0, le=1.0)


# ---------- Step 4: Final output (with provenance) ----------
class FinalAmount(BaseModel):
    type: AmountType
    value: float
    source: str  # e.g. "text: 'Total: INR 1200'"


class FinalOutput(BaseModel):
    currency: Optional[str] = None
    amounts: List[FinalAmount]
    status: Literal["ok", "needs_review"] = "ok"


# ---------- Guardrail / error response ----------
class GuardrailResponse(BaseModel):
    status: Literal["no_amounts_found", "needs_review", "error"]
    reason: str