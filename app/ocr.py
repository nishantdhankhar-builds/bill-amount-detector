import re
from typing import List, Optional, Union

from app.schemas import GuardrailResponse, OCRResult

# Letters that OCR commonly confuses with digits (l->1, O->0, S->5, B->8)
OCR_LOOKALIKES = "OoIlSB"

BASE_CONFIDENCE = 0.95   # confidence for clean typed text
NOISE_PENALTY = 0.15     # max reduction when every token looks OCR-damaged
MIN_CONFIDENCE = 0.40    # below this, we refuse to continue

# A numeric token: starts with a digit or lookalike, may contain , . and may end in %
# The lookarounds stop us from pulling "0" out of a word like "T0tal".
TOKEN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])"
    r"([0-9OoIlSB][0-9OoIlSB.,]*%?)"
    r"(?![A-Za-z0-9])"
)

# "Rs1200" or "INR1200" -> "Rs 1200" so the number becomes its own token
CURRENCY_GLUE = re.compile(r"(?i)(?<![a-z])(inr|rs\.?|₹)(?=\d)")

INR_PATTERN = re.compile(r"(?i)(?<![a-z])(inr|rs\.?|rupees?)(?![a-z])|₹")


def detect_currency(text: str) -> Optional[str]:
    if INR_PATTERN.search(text):
        return "INR"
    if "$" in text or re.search(r"(?i)\busd\b", text):
        return "USD"
    return None


def find_tokens(text: str) -> List[str]:
    prepared = CURRENCY_GLUE.sub(r"\1 ", text)
    tokens = []
    for match in TOKEN_PATTERN.finditer(prepared):
        token = match.group(1).rstrip(".,")
        if any(ch.isdigit() for ch in token):   # must contain at least one real digit
            tokens.append(token)
    return tokens


def extract_tokens(
    text: str, ocr_confidence: float = 1.0
) -> Union[OCRResult, GuardrailResponse]:
    """Step 1: raw text in, raw numeric tokens out (or a guardrail response)."""
    if not text or not text.strip():
        return GuardrailResponse(status="no_amounts_found", reason="empty input")

    tokens = find_tokens(text)
    if not tokens:
        return GuardrailResponse(status="no_amounts_found", reason="document too noisy")

    noisy = sum(1 for t in tokens if any(c in OCR_LOOKALIKES for c in t))
    confidence = round(
        max(0.0, (BASE_CONFIDENCE - NOISE_PENALTY * noisy / len(tokens)) * ocr_confidence), 2
    )
    if confidence < MIN_CONFIDENCE:
        return GuardrailResponse(status="no_amounts_found", reason="document too noisy")

    return OCRResult(
        raw_text=text.strip(),
        raw_tokens=tokens,
        currency_hint=detect_currency(text),
        confidence=confidence,
    )