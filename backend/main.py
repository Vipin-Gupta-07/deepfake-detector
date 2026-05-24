"""
main.py
-------
FastAPI application for Deepfake Image Detection.

Endpoints
---------
GET  /          → health check
POST /predict   → accepts image upload, returns Real/Fake prediction
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from model import load_model, predict, DeepfakeViT
from preprocessing import FacePreprocessor

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared application state
# ---------------------------------------------------------------------------
_state: dict = {}

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
}

MAX_FILE_BYTES = 15 * 1024 * 1024   # 15 MB


# ---------------------------------------------------------------------------
# Lifespan – model and preprocessor are loaded once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== Starting Deepfake Detector API ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    _state["device"]       = device
    _state["model"]        = load_model(device=device)
    _state["preprocessor"] = FacePreprocessor(device=device)

    logger.info("=== API ready ===")
    yield
    logger.info("=== Shutting down ===")
    _state.clear()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Deepfake Image Detection API",
    description=(
        "Detects whether a facial image is real or AI-generated (deepfake) "
        "using a fine-tuned ViT-B/16 Vision Transformer with MTCNN face preprocessing."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS – allow the frontend (any origin in dev; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://vipin-gupta-07.github.io",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    label:       str    # "Real" | "Fake"
    confidence:  float  # probability of the predicted class (%)
    real_prob:   float  # probability of Real (%)
    fake_prob:   float  # probability of Fake (%)
    face_found:  bool   # whether MTCNN detected a face
    latency_ms:  float  # end-to-end processing time


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
async def health():
    """Simple health-check endpoint."""
    return {
        "status": "ok",
        "model":  "ViT-B/16 deepfake detector",
        "device": str(_state.get("device", "not loaded")),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    summary="Classify an image as Real or Fake",
)
async def predict_image(file: UploadFile = File(...)):
    """
    Upload a facial image (JPEG / PNG / WebP / BMP) and receive a
    Real / Fake classification with confidence scores.

    The pipeline mirrors the paper's methodology:
      1. MTCNN face detection → crop + border → 224×224
      2. Fallback to full image if no face is detected
      3. ViT-B/16 binary classification with sigmoid output
    """
    t_start = time.perf_counter()

    # ---- Validate content type ----
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported media type '{file.content_type}'. "
                f"Accepted types: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}"
            ),
        )

    # ---- Read file bytes ----
    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {MAX_FILE_BYTES // (1024*1024)} MB.",
        )
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ---- Preprocess ----
    try:
        preprocessor: FacePreprocessor = _state["preprocessor"]
        tensor, face_found = preprocessor.process(image_bytes)
    except Exception as exc:
        logger.exception("Preprocessing failed")
        raise HTTPException(status_code=422, detail=f"Image preprocessing error: {exc}")

    # ---- Inference ----
    try:
        model: DeepfakeViT    = _state["model"]
        device: torch.device  = _state["device"]
        result = predict(model, tensor, device)
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Model inference error: {exc}")

    latency_ms = round((time.perf_counter() - t_start) * 1000, 1)

    logger.info(
        "file=%s  face_found=%s  label=%s  fake_prob=%.1f%%  latency=%.1fms",
        file.filename,
        face_found,
        result["label"],
        result["fake_prob"],
        latency_ms,
    )

    return PredictionResponse(
        **result,
        face_found=face_found,
        latency_ms=latency_ms,
    )
