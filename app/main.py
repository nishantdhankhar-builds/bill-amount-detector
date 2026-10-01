from typing import Union

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from app.classify import classify_amounts
from app.normalize import normalize_amounts
from app.ocr import extract_tokens, image_to_text
from app.pipeline import run_image_pipeline, run_pipeline
from app.schemas import (
    ClassificationResult,
    FinalOutput,
    GuardrailResponse,
    NormalizedResult,
    OCRResult,
)

MAX_IMAGE_BYTES = 10 * 1024 * 1024   # 10 MB upload limit

app = FastAPI(title="Medical Bill Amount Detector")


class TextInput(BaseModel):
    text: str


async def read_upload(file: UploadFile) -> Union[bytes, GuardrailResponse]:
    data = await file.read()
    if not data:
        return GuardrailResponse(status="error", reason="empty file")
    if len(data) > MAX_IMAGE_BYTES:
        return GuardrailResponse(status="error", reason="file too large (max 10 MB)")
    return data


@app.get("/health")
def health():
    return {"status": "ok"}


# ---- Per-step endpoints (text) ----
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


# ---- Full pipeline ----
@app.post("/extract/text", response_model=Union[FinalOutput, GuardrailResponse])
def extract_text(body: TextInput):
    return run_pipeline(body.text)


@app.post("/extract/image", response_model=Union[FinalOutput, GuardrailResponse])
async def extract_image(file: UploadFile = File(...)):
    data = await read_upload(file)
    if isinstance(data, GuardrailResponse):
        return data
    return run_image_pipeline(data)


# ---- Debug helper: see what Tesseract read ----
@app.post("/ocr/image")
async def ocr_image(file: UploadFile = File(...)):
    data = await read_upload(file)
    if isinstance(data, GuardrailResponse):
        return data
    result = image_to_text(data)
    if isinstance(result, GuardrailResponse):
        return result
    text, confidence = result
    return {"text": text, "ocr_confidence": confidence}