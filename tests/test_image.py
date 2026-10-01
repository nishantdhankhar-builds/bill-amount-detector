from app.pipeline import run_image_pipeline
from app.schemas import GuardrailResponse


def test_invalid_image_triggers_guardrail():
    r = run_image_pipeline(b"this is not an image")
    assert isinstance(r, GuardrailResponse)
    assert r.status == "error"