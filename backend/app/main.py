from __future__ import annotations

import base64
import io
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field, ValidationError
from pycocotools import mask as mask_utils
import torch

try:  # Deferred import so the API can start even if SAM2 is not installed yet.
    from sam2.build_sam2 import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except Exception:  # pragma: no cover - we just want a friendly runtime error later.
    SAM2ImagePredictor = None  # type: ignore[assignment]


class PromptPayload(BaseModel):
    """Payload describing positive/negative clicks and optional boxes."""

    points: list[tuple[float, float]] = Field(default_factory=list, description="Pixel coordinates")
    labels: list[int] = Field(default_factory=list, description="1 = foreground, 0 = background")
    boxes: list[list[float]] | None = Field(default=None, description="[x1, y1, x2, y2] in pixels")
    multimask: bool = True
    top_k: int = Field(default=3, le=5, ge=1)


class MaskPayload(BaseModel):
    score: float
    bbox: list[float]
    area: float
    rle: dict


class SegmentResponse(BaseModel):
    id: str
    file_name: str
    created_at: datetime
    preview_overlay: str
    masks: list[MaskPayload]


OUTPUT_DIR = Path(os.getenv("SAM2_OUTPUT_DIR", "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CONFIG = Path(os.getenv("SAM2_CONFIG_PATH", "weights/sam2.1_hiera_tiny.yaml"))
MODEL_CHECKPOINT = Path(os.getenv("SAM2_CHECKPOINT_PATH", "weights/sam2.1_hiera_tiny.pt"))
DEVICE = os.getenv("SAM2_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")  # type: ignore[name-defined]


_predictor: SAM2ImagePredictor | None = None  # type: ignore[name-defined]


def _lazy_load_predictor() -> SAM2ImagePredictor:
    if SAM2ImagePredictor is None:
        raise RuntimeError("sam2 is not installed. Install it via `pip install segment-anything-2`.")
    global _predictor
    if _predictor is not None:
        return _predictor
    if not MODEL_CONFIG.exists() or not MODEL_CHECKPOINT.exists():
        raise RuntimeError(
            f"SAM2 config ({MODEL_CONFIG}) or checkpoint ({MODEL_CHECKPOINT}) not found. "
            "Download weights from the official release and update the env variables."
        )
    model = build_sam2(config_path=str(MODEL_CONFIG), checkpoint=str(MODEL_CHECKPOINT), device=DEVICE)
    predictor = SAM2ImagePredictor(model)
    predictor.model.eval()
    _predictor = predictor
    return predictor


def _read_image(upload: UploadFile) -> np.ndarray:
    contents = upload.file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    return np.array(pil_image)


def _encode_mask(mask: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    bbox = mask_utils.toBbox(rle).tolist()
    area = float(mask_utils.area(rle))
    return {"rle": rle, "bbox": bbox, "area": area}


def _overlay(image: np.ndarray, mask: np.ndarray) -> str:
    overlay = image.copy()
    color = np.array([30, 144, 255], dtype=np.uint8)
    alpha = 0.45
    overlay[mask > 0] = (alpha * color + (1 - alpha) * overlay[mask > 0]).astype(np.uint8)
    image_file = io.BytesIO()
    Image.fromarray(overlay).save(image_file, format="PNG")
    return "data:image/png;base64," + base64.b64encode(image_file.getvalue()).decode("utf-8")


def _save_artifacts(metadata: SegmentResponse, original: np.ndarray, masks: list[np.ndarray]) -> None:
    sample_dir = OUTPUT_DIR / metadata.id
    sample_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(original).save(sample_dir / "original.png", format="PNG")
    (sample_dir / "meta.json").write_text(metadata.model_dump_json(indent=2))
    for idx, mask in enumerate(masks):
        Image.fromarray(mask.astype(np.uint8) * 255).save(sample_dir / f"mask_{idx}.png", format="PNG")


app = FastAPI(title="SAM2 Segment Anything Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("SAM2_ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.post("/segment", response_model=SegmentResponse)
async def segment_endpoint(image: Annotated[UploadFile, File(...)], payload: Annotated[str, Form(...)]) -> SegmentResponse:
    try:
        prompts = PromptPayload.model_validate_json(payload)
    except ValidationError as exc:  # Propagate structured validation information.
        raise HTTPException(status_code=400, detail=exc.errors()) from exc

    predictor = _lazy_load_predictor()
    image_np = _read_image(image)
    predictor.set_image(image_np)

    point_coords = np.array(prompts.points, dtype=np.float32) if prompts.points else None
    point_labels = np.array(prompts.labels, dtype=np.int32) if prompts.labels else None
    if (point_coords is None) != (point_labels is None):
        raise HTTPException(status_code=400, detail="points and labels must be provided together")

    if prompts.boxes:
        boxes = np.array(prompts.boxes, dtype=np.float32)
    else:
        boxes = None

    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box_coords=boxes,
        multimask_output=prompts.multimask,
    )
    if masks is None or not len(masks):
        raise HTTPException(status_code=422, detail="SAM2 returned no masks")

    order = np.argsort(np.array(scores))[::-1][: prompts.top_k]
    ordered_masks = [masks[idx] for idx in order]
    ordered_scores = [float(scores[idx]) for idx in order]

    overlay_image = _overlay(image_np, ordered_masks[0])
    masks_payload = []
    for mask, score in zip(ordered_masks, ordered_scores):
        encoded = _encode_mask(mask)
        masks_payload.append(
            MaskPayload(score=score, bbox=encoded["bbox"], area=encoded["area"], rle=encoded["rle"])
        )

    response = SegmentResponse(
        id=str(uuid.uuid4()),
        file_name=image.filename or "upload.png",
        created_at=datetime.utcnow(),
        preview_overlay=overlay_image,
        masks=masks_payload,
    )
    _save_artifacts(response, image_np, ordered_masks)
    return response


@app.get("/segment/{sample_id}", response_model=SegmentResponse)
async def get_segmentation(sample_id: str) -> SegmentResponse:
    meta_path = OUTPUT_DIR / sample_id / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Segmentation not found")
    payload = json.loads(meta_path.read_text())
    return SegmentResponse(**payload)


@app.get("/healthz")
async def health() -> dict[str, str]:
    return {"status": "ok"}
