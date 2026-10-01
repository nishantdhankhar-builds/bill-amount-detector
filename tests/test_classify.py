from app.classify import classify_amounts
from app.normalize import normalize_amounts
from app.ocr import extract_tokens
from app.schemas import ClassificationResult, GuardrailResponse


def run(text):
    step1 = extract_tokens(text)
    step2 = normalize_amounts(step1)
    return classify_amounts(step1, step2)


def as_pairs(result):
    return [(a.type, a.value) for a in result.amounts]


def test_clean_bill():
    r = run("Total: INR 1200 | Paid: 1000 | Due: 200 | Discount: 10%")
    assert isinstance(r, ClassificationResult)
    assert as_pairs(r) == [("total_bill", 1200), ("paid", 1000), ("due", 200)]
    assert r.amounts[0].source == "Total: INR 1200"


def test_noisy_bill():
    r = run("T0tal: Rs l200 | Pald: 1000 | Due: 200")
    assert isinstance(r, ClassificationResult)
    assert as_pairs(r) == [("total_bill", 1200), ("paid", 1000), ("due", 200)]
    assert r.confidence < 0.95


def test_no_separators():
    r = run("Total 1200 Paid 1000 Due 200")
    assert as_pairs(r) == [("total_bill", 1200), ("paid", 1000), ("due", 200)]


def test_unlabeled_triggers_guardrail():
    r = run("Item 500")
    assert isinstance(r, GuardrailResponse)
    assert r.status == "needs_review"