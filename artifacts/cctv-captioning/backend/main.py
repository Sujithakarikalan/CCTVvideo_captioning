"""FastAPI adapter for the existing Phase 6/7 CCTV captioning project."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ARTIFACT_ROOT / "project"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(512 * 1024 * 1024)))
ALLOWED_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
logger = logging.getLogger("cctv-captioning")

app = FastAPI(title="CCTV Video Captioning API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@lru_cache(maxsize=1)
def get_captioner() -> Any:
    """Load the existing Qwen model once, on the first analysis request."""
    from src.models.qwen3_video_captioner import Qwen3VideoCaptioner

    return Qwen3VideoCaptioner(
        device="auto",
        model_name=MODEL_NAME,
        cache_dir=PROJECT_ROOT / "checkpoints" / "qwen3_vl",
    )


def _device_status() -> str:
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "unavailable"


def _run_yolo_if_available(video_path: Path) -> tuple[list[dict[str, Any]], str]:
    """Use the existing YOLO wrapper only when project-local weights are present."""
    weights = PROJECT_ROOT / "checkpoints" / "yolo" / "yolov8n.pt"
    if not weights.is_file():
        return [], ""

    try:
        from src.data.video_io import iter_sampled_frames
        from src.features.caption_evidence import format_caption_support, summarize_yolo_hints
        from src.models.yolo_extractor import YOLODetectionExtractor, detections_to_dict

        frames = list(iter_sampled_frames(video_path, sample_fps=1.0, max_frames=16))
        detections = YOLODetectionExtractor(weights=weights, device="auto").detect_frames(
            [frame.frame_bgr for frame in frames]
        )
        frame_records = [{"detections": detections_to_dict(items)} for items in detections]
        hints, _metadata = summarize_yolo_hints(frame_records)
        return hints, format_caption_support(hints, [])
    except Exception:
        logger.exception("Optional YOLO evidence failed; continuing with Qwen video inference")
        return [], ""


def _analyze_saved_video(video_path: Path, video_filename: str) -> dict[str, Any]:
    from src.data.video_io import get_video_metadata
    from src.features.caption_evidence import summarize_yolo_hints

    try:
        metadata = get_video_metadata(video_path)
    except Exception as exc:
        raise ValueError("The uploaded video could not be decoded by OpenCV.") from exc

    requested_frames = min(16, metadata.total_frames) if metadata.total_frames > 0 else 16
    yolo_hints, supporting_text = _run_yolo_if_available(video_path)
    generation = get_captioner().caption(
        video_path,
        num_frames=requested_frames,
        max_new_tokens=128,
        supporting_text=supporting_text,
    )
    objects = [
        {
            "label": item["label"],
            "detection_hits_across_frames": item["detection_hits_across_frames"],
            "sampled_frames_present": item["sampled_frames_present"],
            "sampled_frame_count": item["sampled_frame_count"],
            "max_confidence": item["max_confidence"],
            "interpretation": item["interpretation"],
        }
        for item in yolo_hints
    ]
    return {
        "success": True,
        "video_filename": video_filename,
        "caption": generation.caption,
        "model": generation.model_name,
        "device": generation.device,
        "inference_seconds": generation.inference_seconds,
        "num_frames": generation.num_frames,
        "duration_seconds": metadata.duration_seconds,
        "fps": metadata.fps,
        "width": metadata.width,
        "height": metadata.height,
        "objects": objects,
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": _device_status(),
        "model_loaded": get_captioner.cache_info().currsize > 0,
    }


@app.post("/api/analyze")
async def analyze(video: UploadFile = File(...)) -> dict[str, Any]:
    if not video.filename:
        raise HTTPException(status_code=400, detail="Choose a video file to analyze.")

    suffix = Path(video.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format. Upload an MP4, AVI, MOV, or MKV file.",
        )

    safe_filename = Path(video.filename.replace("\\", "/")).name
    temp_path: Path | None = None
    total_bytes = 0
    try:
        with tempfile.NamedTemporaryFile(prefix="cctv-upload-", suffix=suffix, delete=False) as handle:
            temp_path = Path(handle.name)
            while chunk := await video.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Video exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
                    )
                handle.write(chunk)

        if total_bytes == 0:
            raise HTTPException(status_code=400, detail="The uploaded video is empty.")

        return await asyncio.to_thread(_analyze_saved_video, temp_path, safe_filename)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Video caption inference failed")
        raise HTTPException(
            status_code=500,
            detail="Qwen3-VL inference failed. Check backend logs and confirm the model/GPU environment is available.",
        ) from exc
    finally:
        await video.close()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
