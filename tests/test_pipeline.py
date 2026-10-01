from app.pipeline import run_pipeline
from app.schemas import FinalOutput, GuardrailResponse


def test_clean_bill_matches_assignment_format():
    r = run_pipeline("Total: INR 1200 | Paid: 1000 | Due: 200 | Discount: 10%")
    assert isinstance(r, FinalOutput)
    assert r.currency == "INR"
    assert r.status == "ok"
    assert [(a.type, a.value) for a in r.amounts] == [
        ("total_bill", 1200), ("paid", 1000), ("due", 200)
    ]
    assert r.amounts[0].source == "text: 'Total: INR 1200'"


def test_noisy_bill_still_ok():
    r = run_pipeline("T0tal: Rs l200 | Pald: 1000 | Due: 200")
    assert isinstance(r, FinalOutput)
    assert r.status == "ok"
    assert r.amounts[0].value == 1200


def test_arithmetic_mismatch_needs_review():
    r = run_pipeline("Total: INR 1200 | Paid: 1000 | Due: 300")
    assert isinstance(r, FinalOutput)
    assert r.status == "needs_review"
    assert any("does not equal" in w for w in r.warnings)


def test_guardrail_passthrough():
    assert isinstance(run_pipeline("hello"), GuardrailResponse)


def test_at_sign_misread_as_zero():
    r = run_pipeline("Total: INR 12@@ | Paid: 10@@ | Due: 20@ | Discount: 10%")
    assert isinstance(r, FinalOutput)
    assert [(a.type, a.value) for a in r.amounts] == [
        ("total_bill", 1200), ("paid", 1000), ("due", 200)
    ]