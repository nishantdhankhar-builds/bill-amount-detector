import re
from typing import List, Optional, Tuple, Union

from app.schemas import (
    GuardrailResponse,
    NormalizedResult,
    NormalizedToken,
    OCRResult,
)

# Letters OCR often confuses with digits
DIGIT_FIXES = {"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "B": "8"}

BASE_CONFIDENCE = 0.95
CORRECTION_PENALTY = 0.20   # max reduction when every token needed fixing

VALID_NUMBER = re.compile(r"^\d[\d.,]*$")
THOUSANDS_COMMAS = re.compile(r"^\d{1,3}(,\d{2,3})*,\d{3}$")   # 1,200 / 1,00,000 / 1,200,000
THOUSANDS_DOTS = re.compile(r"^\d{1,3}(\.\d{3})+$")            # 1.200.000 (2+ groups only)


def fix_ocr_digits(token: str) -> Tuple[str, bool]:
    """Replace look-alike letters with digits. Returns (fixed_token, was_changed)."""
    fixed = "".join(DIGIT_FIXES.get(ch, ch) for ch in token)
    return fixed, fixed != token


def parse_number(s: str) -> Optional[float]:
    """Parse '1,200.00', '1.200,00', '1,00,000', '12,50' into a float."""
    if not VALID_NUMBER.match(s):
        return None

    if "," in s and "." in s:
        # whichever separator comes last is the decimal point
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")    # 1.200,00
        else:
            s = s.replace(",", "")                       # 1,200.00
    elif "," in s:
        if THOUSANDS_COMMAS.match(s):
            s = s.replace(",", "")                       # 1,200
        else:
            s = s.replace(",", ".")                      # 12,50 -> 12.50
    elif "." in s and THOUSANDS_DOTS.match(s):
        s = s.replace(".", "")                           # 1.200.000

    try:
        return float(s)
    except ValueError:
        return None


def normalize_amounts(ocr: OCRResult) -> Union[NormalizedResult, GuardrailResponse]:
    """Step 2: raw tokens in, clean numbers out (or a guardrail response)."""
    items: List[NormalizedToken] = []

    for raw in ocr.raw_tokens:
        is_percent = raw.endswith("%")
        body = raw.rstrip("%")
        fixed, corrected = fix_ocr_digits(body)
        value = parse_number(fixed)
        if value is None:
            continue    # unparseable token: drop it rather than guess
        items.append(
            NormalizedToken(raw=raw, value=value, is_percent=is_percent, corrected=corrected)
        )

    amounts = [t.value for t in items if not t.is_percent]
    percentages = [t.value for t in items if t.is_percent]

    if not amounts:
        return GuardrailResponse(
            status="no_amounts_found", reason="no valid amounts after normalization"
        )

    corrected_fraction = sum(t.corrected for t in items) / len(items)
    confidence = BASE_CONFIDENCE - CORRECTION_PENALTY * corrected_fraction
    confidence = round(min(confidence, ocr.confidence), 2)   # never exceed Step 1's confidence

    return NormalizedResult(
        normalized_amounts=amounts,
        percentages=percentages,
        tokens=items,
        normalization_confidence=confidence,
    )