from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

import main


client = TestClient(main.app)


def test_health_reports_backend_model_and_device(monkeypatch):
    monkeypatch.setattr(main, "_device_status", lambda: "cpu")
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model": "Qwen/Qwen3-VL-4B-Instruct",
        "device": "cpu",
        "model_loaded": False,
    }


def test_rejects_unsupported_upload():
    response = client.post(
        "/api/analyze",
        files={"video": ("clip.txt", b"not a video", "text/plain")},
    )
    assert response.status_code == 400
    assert "Unsupported video format" in response.json()["detail"]


def test_rejects_empty_upload():
    response = client.post(
        "/api/analyze",
        files={"video": ("clip.mp4", b"", "video/mp4")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "The uploaded video is empty."


def test_upload_passes_temporary_file_to_existing_captioner_and_returns_actual_result(
    monkeypatch,
):
    passed_path: Path | None = None

    class StubCaptioner:
        def caption(self, path, **kwargs):
            nonlocal passed_path
            passed_path = Path(path)
            assert passed_path.is_file()
            assert kwargs["num_frames"] == 4
            assert kwargs["supporting_text"] == ""
            return SimpleNamespace(
                caption="A delivery van moves through the camera view.",
                model_name="Qwen/Qwen3-VL-4B-Instruct",
                device="cpu",
                num_frames=4,
                inference_seconds=2.75,
            )

    monkeypatch.setattr(
        main,
        "get_captioner",
        lambda: StubCaptioner(),
    )
    monkeypatch.setattr(
        main,
        "_run_yolo_if_available",
        lambda _path: ([], ""),
    )
    monkeypatch.setattr(
        "src.data.video_io.get_video_metadata",
        lambda _path: SimpleNamespace(
            total_frames=4,
            fps=2.0,
            duration_seconds=2.0,
            width=320,
            height=240,
        ),
    )

    response = client.post(
        "/api/analyze",
        files={"video": ("new-clip.mp4", b"uploaded test video bytes", "video/mp4")},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result == {
        "success": True,
        "video_filename": "new-clip.mp4",
        "caption": "A delivery van moves through the camera view.",
        "model": "Qwen/Qwen3-VL-4B-Instruct",
        "device": "cpu",
        "inference_seconds": 2.75,
        "num_frames": 4,
        "duration_seconds": 2.0,
        "fps": 2.0,
        "width": 320,
        "height": 240,
        "objects": [],
    }
    assert passed_path is not None
    assert not passed_path.exists()


def test_inference_errors_are_reported_without_stack_traces(monkeypatch):
    def fail_analysis(_path, _filename):
        raise RuntimeError("internal model details")

    monkeypatch.setattr(main, "_analyze_saved_video", fail_analysis)
    response = client.post(
        "/api/analyze",
        files={"video": ("new-clip.mp4", b"video bytes", "video/mp4")},
    )
    assert response.status_code == 500
    assert "Qwen3-VL inference failed" in response.json()["detail"]
    assert "internal model details" not in response.text
