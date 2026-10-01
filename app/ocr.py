import io
import os
import re
from typing import List, Optional, Tuple, Union

import pytesseract
from PIL import Image, ImageOps, UnidentifiedImageError

from app.schemas import GuardrailResponse, OCRResult

# Tell pytesseract where Tesseract lives on Windows (skipped if not there)
WIN_TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(WIN_TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = WIN_TESSERACT_PATH

# Characters that OCR commonly confuses with digits (l->1, O->0, S->5, B->8, @->0)
OCR_LOOKALIKES = "OoIlSB@"

BASE_CONFIDENCE = 0.95   # confidence for clean typed text
NOISE_PENALTY = 0.15     # max reduction when every token looks OCR-damaged
MIN_CONFIDENCE = 0.40    # below this, we refuse to continue
MIN_IMAGE_WIDTH = 1000   # small images are upscaled before OCR

TOKEN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])"
    r"([0-9OoIlSB@][0-9OoIlSB@.,]*%?)"
    r"(?![A-Za-z0-9])"
)

CURRENCY_GLUE = re.compile(r"(?i)(?<![a-z])(inr|rs\.?|₹)(?=\d)")

INR_PATTERN = re.compile(r"(?i)(?<![a-z])(inr|rs\.?|rupees?)(?![a-z])|₹")


def detect_currency(text: str) -> Optional[str]:
    if INR_PATTERN.search(text):
        return "INR"
    if "$" in text or re.search(r"(?i)\busd\b", text):
        return "USD"
    return None


def find_tokens(text: str) -> List[str]:
    prepared = CURRENCY_GLUE.sub(r"\1 ", text)
    tokens = []
    for match in TOKEN_PATTERN.finditer(prepared):
        token = match.group(1).rstrip(".,")
        if any(ch.isdigit() for ch in token):
            tokens.append(token)
    return tokens


def extract_tokens(
    text: str, ocr_confidence: float = 1.0
) -> Union[OCRResult, GuardrailResponse]:
    """Step 1: raw text in, raw numeric tokens out (or a guardrail response)."""
    if not text or not text.strip():
        return GuardrailResponse(status="no_amounts_found", reason="empty input")

    tokens = find_tokens(text)
    if not tokens:
        return GuardrailResponse(status="no_amounts_found", reason="document too noisy")

    noisy = sum(1 for t in tokens if any(c in OCR_LOOKALIKES for c in t))
    confidence = round(
        max(0.0, (BASE_CONFIDENCE - NOISE_PENALTY * noisy / len(tokens)) * ocr_confidence), 2
    )
    if confidence < MIN_CONFIDENCE:
        return GuardrailResponse(status="no_amounts_found", reason="document too noisy")

    return OCRResult(
        raw_text=text.strip(),
        raw_tokens=tokens,
        currency_hint=detect_currency(text),
        confidence=confidence,
    )


def image_to_text(image_bytes: bytes) -> Union[Tuple[str, float], GuardrailResponse]:
    """Image in, (text, tesseract_confidence 0-1) out. Lines are kept as separate lines."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except (UnidentifiedImageError, OSError):
        return GuardrailResponse(status="error", reason="invalid or unreadable image")

    # Preprocess: fix rotation from phone photos, grayscale, boost contrast, upscale
    img = ImageOps.exif_transpose(img).convert("L")
    img = ImageOps.autocontrast(img)
    if img.width < MIN_IMAGE_WIDTH:
        scale = MIN_IMAGE_WIDTH / img.width
        img = img.resize((MIN_IMAGE_WIDTH, int(img.height * scale)))

    try:
        data = pytesseract.image_to_data(
            img, config="--psm 6", output_type=pytesseract.Output.DICT
        )
    except pytesseract.TesseractNotFoundError:
        return GuardrailResponse(status="error", reason="Tesseract OCR engine not found")

    # Rebuild text line by line, and average the per-word confidence
    lines = {}
    confs = []
    for i, word in enumerate(data["text"]):
        word = word.strip()
        conf = float(data["conf"][i])
        if not word or conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
        confs.append(conf)

    if not lines:
        return GuardrailResponse(status="no_amounts_found", reason="no text detected in image")

    text = "\n".join(" ".join(words) for _, words in sorted(lines.items()))
    return text, round(sum(confs) / len(confs) / 100, 2)