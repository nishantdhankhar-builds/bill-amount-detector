from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

from app.classify import classify_amounts
from app.normalize import normalize_amounts
from app.ocr import extract_tokens
from app.schemas import (
    ClassificationResult,
    GuardrailResponse,
    NormalizedResult,
    OCRResult,
)

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
        return step1
    return normalize_amounts(step1)


@app.post("/classify/text", response_model=Union[ClassificationResult, GuardrailResponse])
def classify_text(body: TextInput):
    step1 = extract_tokens(body.text)
    if isinstance(step1, GuardrailResponse):
        return step1
    step2 = normalize_amounts(step1)
    if isinstance(step2, GuardrailResponse):
        return step2
    return classify_amounts(step1, step2)