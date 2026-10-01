import difflib
import re
from typing import List, Optional, Tuple, Union

from app.schemas import (
    ClassificationResult,
    ClassifiedAmount,
    GuardrailResponse,
    NormalizedResult,
    OCRResult,
)

# Order matters: "subtotal" is checked before "total"
KEYWORDS = {
    "subtotal": ["subtotal", "sub total"],
    "total_bill": ["total", "grand total", "total bill", "bill amount", "net payable", "amount payable"],
    "paid": ["paid", "amount paid", "payment received", "received", "advance"],
    "due": ["due", "balance", "outstanding", "pending", "remaining"],
    "discount": ["discount", "concession", "waiver"],
    "tax": ["tax", "gst", "cgst", "sgst"],
}

EXACT_SCORE = 1.0
FUZZY_SCORE = 0.85       # label needed OCR-tolerant matching
UNLABELED_SCORE = 0.40   # no keyword found
FUZZY_CUTOFF = 0.70      # how similar a word must be to a keyword

SEPARATORS = "|\n;"

# single-word keywords for fuzzy matching
_SINGLE_WORDS = {kw: t for t, kws in KEYWORDS.items() for kw in kws if " " not in kw}


def match_label(region: str) -> Tuple[Optional[str], float]:
    """Return (amount_type, score) for the words in a text region."""
    text = region.lower()

    # 1) exact keyword or phrase, as whole words
    for amount_type, keywords in KEYWORDS.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", text):
                return amount_type, EXACT_SCORE

    # 2) fuzzy match per word, to survive OCR damage like "Pald" or "T0tal"
    for word in re.findall(r"[a-z0-9]+", text):
        if len(word) < 3:
            continue
        close = difflib.get_close_matches(word, _SINGLE_WORDS.keys(), n=1, cutoff=FUZZY_CUTOFF)
        if close:
            return _SINGLE_WORDS[close[0]], FUZZY_SCORE

    return None, UNLABELED_SCORE


def classify_amounts(
    ocr: OCRResult, norm: NormalizedResult
) -> Union[ClassificationResult, GuardrailResponse]:
    """Step 3: label each rupee amount using the words around it."""
    text = ocr.raw_text
    labeled: List[ClassifiedAmount] = []
    scores: List[float] = []
    cursor = 0      # where we search for the next token
    prev_end = 0    # end of the previous token, so labels never leak across numbers

    for tok in norm.tokens:
        idx = text.find(tok.raw, cursor)
        if idx == -1:
            continue
        end = idx + len(tok.raw)
        cursor = end

        if tok.is_percent:
            prev_end = end
            continue    # percentages are not rupee amounts

        # region before the number: back to the last separator or previous number
        last_sep = max(text.rfind(s, 0, idx) for s in SEPARATORS)
        start = max(prev_end, last_sep + 1)
        before = text[start:idx]

        amount_type, score = match_label(before)

        if amount_type is None:
            # fall back to words after the number, up to the next separator
            rest = text[end:]
            stops = [rest.find(s) for s in SEPARATORS if s in rest]
            after = rest[: min(stops)] if stops else rest
            amount_type, score = match_label(after)

        prev_end = end
        source = text[start:end].strip(" \t") or tok.raw
        labeled.append(
            ClassifiedAmount(type=amount_type or "other", value=tok.value, source=source)
        )
        scores.append(score)

    if not labeled or all(a.type == "other" for a in labeled):
        return GuardrailResponse(
            status="needs_review", reason="could not label any amount from its context"
        )

    confidence = round(norm.normalization_confidence * sum(scores) / len(scores), 2)
    return ClassificationResult(amounts=labeled, confidence=confidence)