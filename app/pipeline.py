from typing import Dict, List, Union

from app.classify import classify_amounts
from app.normalize import normalize_amounts
from app.ocr import extract_tokens, image_to_text
from app.schemas import (
    ClassifiedAmount,
    ClassificationResult,
    FinalAmount,
    FinalOutput,
    GuardrailResponse,
    OCRResult,
)

TOLERANCE = 0.01             # allowed rounding difference in paid + due = total
REVIEW_CONFIDENCE = 0.60     # below this, a human should look at the result


def check_consistency(amounts: List[ClassifiedAmount]) -> List[str]:
    """Sanity checks on the labeled amounts. Returns a list of warnings."""
    warnings: List[str] = []
    by_type: Dict[str, List[float]] = {}
    for a in amounts:
        by_type.setdefault(a.type, []).append(a.value)

    for t in ("total_bill", "paid", "due"):
        if len(by_type.get(t, [])) > 1:
            warnings.append(f"multiple '{t}' amounts found")

    if all(t in by_type for t in ("total_bill", "paid", "due")):
        total, paid, due = by_type["total_bill"][0], by_type["paid"][0], by_type["due"][0]
        if abs(paid + due - total) > TOLERANCE:
            warnings.append(f"paid ({paid:g}) + due ({due:g}) does not equal total ({total:g})")

    if "other" in by_type:
        warnings.append("some amounts could not be labeled")

    return warnings


def build_final_output(
    ocr: OCRResult, classified: ClassificationResult, source_kind: str = "text"
) -> FinalOutput:
    """Step 4: combine everything, add provenance and the review status."""
    warnings = check_consistency(classified.amounts)
    if classified.confidence < REVIEW_CONFIDENCE:
        warnings.append(f"low confidence ({classified.confidence})")

    amounts = [
        FinalAmount(
            type=a.type,
            value=a.value,
            source=f"{source_kind}: '{a.source}'",
            raw_source=a.raw_source if a.raw_source != a.source else None,
        )
        for a in classified.amounts
    ]
    return FinalOutput(
        currency=ocr.currency_hint,
        amounts=amounts,
        status="needs_review" if warnings else "ok",
        confidence=classified.confidence,
        warnings=warnings,
    )


def run_pipeline(
    text: str, source_kind: str = "text", ocr_confidence: float = 1.0
) -> Union[FinalOutput, GuardrailResponse]:
    """Runs Steps 1-4. Stops at the first guardrail."""
    step1 = extract_tokens(text, ocr_confidence)
    if isinstance(step1, GuardrailResponse):
        return step1

    step2 = normalize_amounts(step1)
    if isinstance(step2, GuardrailResponse):
        return step2

    step3 = classify_amounts(step1, step2)
    if isinstance(step3, GuardrailResponse):
        return step3

    return build_final_output(step1, step3, source_kind)


def run_image_pipeline(image_bytes: bytes) -> Union[FinalOutput, GuardrailResponse]:
    """OCR an image, then run the same Steps 1-4 on the text it contains."""
    result = image_to_text(image_bytes)
    if isinstance(result, GuardrailResponse):
        return result
    text, ocr_confidence = result
    return run_pipeline(text, source_kind="ocr", ocr_confidence=ocr_confidence)