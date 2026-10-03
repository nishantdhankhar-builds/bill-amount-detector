from app import classify
from app.llm import parse_labels
from app.pipeline import run_pipeline


def test_parse_labels_accepts_valid_reply():
    raw = '```json\n{"labels": {"0": "total_bill", "1": "paid"}}\n```'
    assert parse_labels(raw, [0, 1]) == {0: "total_bill", 1: "paid"}


def test_parse_labels_drops_invalid_entries():
    raw = '{"labels": {"0": "total_bill", "7": "paid", "1": "banana", "x": "due"}}'
    assert parse_labels(raw, [0, 1]) == {0: "total_bill"}


def test_parse_labels_survives_garbage():
    assert parse_labels("not json at all", [0]) == {}
    assert parse_labels('{"labels": "oops"}', [0]) == {}


def test_llm_fills_unlabeled_amount(monkeypatch):
    monkeypatch.setattr(classify, "llm_label_tokens", lambda text, items: {0: "total_bill"})
    r = run_pipeline("Net amount 1200 | Advance 1000 | Outstanding 200")
    assert [(a.type, a.value, a.labeled_by) for a in r.amounts] == [
        ("total_bill", 1200, "llm"),
        ("paid", 1000, "rules"),
        ("due", 200, "rules"),
    ]
    assert r.status == "ok"


def test_llm_cannot_touch_rule_labels(monkeypatch):
    # LLM wrongly claims index 1 (already labeled "paid" by rules) is a total
    monkeypatch.setattr(
        classify, "llm_label_tokens", lambda text, items: {0: "total_bill", 1: "total_bill"}
    )
    r = run_pipeline("Net amount 1200 | Advance 1000 | Outstanding 200")
    assert r.amounts[1].type == "paid"
    assert r.amounts[1].labeled_by == "rules"