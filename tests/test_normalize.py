from app.normalize import normalize_amounts, parse_number
from app.schemas import GuardrailResponse, NormalizedResult, OCRResult


def make_ocr(tokens, confidence=0.95):
    return OCRResult(
        raw_text=" ".join(tokens),
        raw_tokens=tokens,
        currency_hint="INR",
        confidence=confidence,
    )


def test_clean_tokens():
    r = normalize_amounts(make_ocr(["1200", "1000", "200", "10%"]))
    assert isinstance(r, NormalizedResult)
    assert r.normalized_amounts == [1200, 1000, 200]
    assert r.percentages == [10]
    assert r.normalization_confidence == 0.95


def test_ocr_digit_fix():
    r = normalize_amounts(make_ocr(["l200", "1000", "200"]))
    assert isinstance(r, NormalizedResult)
    assert r.normalized_amounts == [1200, 1000, 200]
    assert r.tokens[0].corrected is True
    assert r.normalization_confidence < 0.95


def test_number_formats():
    assert parse_number("1,200.00") == 1200
    assert parse_number("1.200,00") == 1200
    assert parse_number("1,00,000") == 100000
    assert parse_number("12,50") == 12.5
    assert parse_number("1.2.3") is None


def test_only_percentage_triggers_guardrail():
    r = normalize_amounts(make_ocr(["10%"]))
    assert isinstance(r, GuardrailResponse)
    assert r.status == "no_amounts_found"


def test_confidence_capped_by_step1():
    r = normalize_amounts(make_ocr(["1200"], confidence=0.60))
    assert r.normalization_confidence == 0.60