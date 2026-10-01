from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

from app.ocr import extract_tokens
from app.schemas import GuardrailResponse, OCRResult

app = FastAPI(title="Medical Bill Amount Detector")


class TextInput(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ocr/text", response_model=Union[OCRResult, GuardrailResponse])
def ocr_text(body: TextInput):
    return extract_tokens(body.text)