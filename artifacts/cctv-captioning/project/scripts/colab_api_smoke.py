"""Run a real, uncached Qwen3-VL caption through the FastAPI upload endpoint.

Run from the exported workspace root in a CUDA-enabled Google Colab runtime:
  python artifacts/cctv-captioning/project/scripts/colab_api_smoke.py --video /content/clip.mp4
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx

ARTIFACT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ARTIFACT_ROOT / "project"
BACKEND_DIR = ARTIFACT_ROOT / "backend"
MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _tail(path: Path, limit: int = 35) -> str:
    if not path.exists():
        return "(backend log file was not created)"
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:])


def _require_cuda() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is missing. Run this check in the CUDA-enabled Colab runtime and "
            "keep Colab's existing CUDA PyTorch installation."
        ) from exc
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available to PyTorch. Select the Colab T4 GPU runtime, restart "
            "the session, and rerun this check; CPU inference is not substituted."
        )
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")


def verify(video_path: Path) -> dict[str, Any]:
    _require_cuda()
    video_path = video_path.expanduser().resolve()
    if not video_path.is_file():
        raise SystemExit(f"Video file does not exist: {video_path}")
    if video_path.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv"}:
        raise SystemExit("Use an MP4, AVI, MOV, or MKV video.")
    if not PROJECT_ROOT.is_dir() or not BACKEND_DIR.is_dir():
        raise SystemExit(
            "The exported workspace is incomplete. Keep artifacts/cctv-captioning/"
            "project and backend together."
        )

    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(PROJECT_ROOT), existing_pythonpath) if part
    )
    log_handle = tempfile.NamedTemporaryFile(
        prefix="cctv-fastapi-colab-", suffix=".log", delete=False
    )
    log_path = Path(log_handle.name)
    log_handle.close()
    with log_path.open("w", encoding="utf-8") as log_stream:
        backend = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "main:app",
                "--app-dir",
                str(BACKEND_DIR),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=ARTIFACT_ROOT,
            env=env,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
        )

    try:
        deadline = time.monotonic() + 120
        health: dict[str, Any] | None = None
        with httpx.Client(timeout=httpx.Timeout(10.0, connect=3.0)) as client:
            while time.monotonic() < deadline:
                if backend.poll() is not None:
                    raise RuntimeError(
                        "FastAPI exited before becoming ready:\n" + _tail(log_path)
                    )
                try:
                    response = client.get(f"{base_url}/api/health")
                    if response.status_code == 200:
                        health = response.json()
                        break
                except httpx.RequestError:
                    time.sleep(1)

            if health is None:
                raise RuntimeError(
                    "FastAPI did not become ready within 120 seconds:\n" + _tail(log_path)
                )
            if health.get("model") != MODEL_NAME or not str(health.get("device", "")).startswith("cuda"):
                raise RuntimeError(
                    "Backend health did not confirm the expected Qwen model and CUDA device: "
                    + json.dumps(health)
                )

            with video_path.open("rb") as video:
                response = client.post(
                    f"{base_url}/api/analyze",
                    files={
                        "video": (
                            video_path.name,
                            video,
                            mimetypes.guess_type(video_path.name)[0] or "application/octet-stream",
                        )
                    },
                    timeout=httpx.Timeout(1800.0, connect=10.0),
                )

            if response.status_code != 200:
                raise RuntimeError(
                    f"POST /api/analyze returned HTTP {response.status_code}: "
                    f"{response.text}\n\nBackend log:\n{_tail(log_path)}"
                )

            result = response.json()
            if (
                result.get("success") is not True
                or result.get("model") != MODEL_NAME
                or not str(result.get("device", "")).startswith("cuda")
                or not isinstance(result.get("caption"), str)
                or not result["caption"].strip()
                or int(result.get("num_frames", 0)) < 1
            ):
                raise RuntimeError(
                    "The live endpoint response did not confirm real Qwen CUDA inference: "
                    + json.dumps(result)
                )

            ready_after = client.get(f"{base_url}/api/health").json()
            if ready_after.get("model_loaded") is not True:
                raise RuntimeError(
                    "The request returned, but the backend health check did not report "
                    "the loaded model."
                )

            print("REAL_QWEN3_VL_API_TEST=PASS")
            print(f"video: {result['video_filename']}")
            print(f"model: {result['model']}")
            print(f"device: {result['device']}")
            print(f"frames: {result['num_frames']}")
            print(f"inference_seconds: {result['inference_seconds']}")
            print(f"caption: {result['caption']}")
            return result
    finally:
        backend.terminate()
        try:
            backend.wait(timeout=10)
        except subprocess.TimeoutExpired:
            backend.kill()
            backend.wait(timeout=5)
        log_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path, help="Real CCTV test video")
    args = parser.parse_args()
    verify(args.video)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
