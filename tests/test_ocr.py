from app.ocr import extract_tokens
from app.schemas import GuardrailResponse, OCRResult


def test_clean_text():
    r = extract_tokens("Total: INR 1200 | Paid: 1000 | Due: 200 | Discount: 10%")
    assert isinstance(r, OCRResult)
    assert r.raw_tokens == ["1200", "1000", "200", "10%"]
    assert r.currency_hint == "INR"
    assert r.confidence > 0.9


def test_noisy_text():
    r = extract_tokens("T0tal: Rs l200 | Pald: 1000 | Due: 200")
    assert isinstance(r, OCRResult)
    assert r.raw_tokens == ["l200", "1000", "200"]
    assert r.currency_hint == "INR"
    assert r.confidence < 0.95


def test_no_amounts():
    r = extract_tokens("Thank you for visiting")
    assert isinstance(r, GuardrailResponse)
    assert r.status == "no_amounts_found"


def test_empty_input():
    assert isinstance(extract_tokens("   "), GuardrailResponse)