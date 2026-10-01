from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

from app.normalize import normalize_amounts
from app.ocr import extract_tokens
from app.schemas import GuardrailResponse, NormalizedResult, OCRResult

app = FastAPI(title="Medical Bill Amount Detector")


class TextInput(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ocr/text", response_model=Union[OCRResult, GuardrailResponse])
def ocr_text(body: TextInput):
    return extract_tokens(body.text)


@app.post("/normalize/text", response_model=Union[NormalizedResult, GuardrailResponse])
def normalize_text(body: TextInput):
    step1 = extract_tokens(body.text)
    if isinstance(step1, GuardrailResponse):
        return step1                      # stop the chain at the first guardrail
    return normalize_amounts(step1)