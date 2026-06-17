"""WebSocket /stream loopback: must match /predict for the same frame.

Gated — needs torch + weights + the FastAPI TestClient stack (httpx). Self-skips
otherwise, like the other live tests."""
from __future__ import annotations

import numpy as np
import pytest


def _test_client_or_skip():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from online_face.serve import create_app
    try:
        app = create_app("retinaface", device="auto")
    except Exception as e:  # offline / download failure
        pytest.skip(f"weights unavailable: {e}")
    return TestClient(app)


@pytest.mark.timeout(300)
def test_ws_stream_matches_predict():
    from online_face._wire import encode_image

    client = _test_client_or_skip()
    frame = (np.random.rand(360, 640, 3) * 255).astype("uint8")
    data, ct = encode_image(frame, "jpeg", 90)

    ref = client.post("/predict", files={"frame": ("f.jpg", data, ct)}).json()["outputs"]
    with client.websocket_connect("/stream") as ws:
        ws.send_bytes(data)
        out = ws.receive_json()["outputs"]

    assert out["shape"] == ref["shape"]
    assert out["boxes"] == ref["boxes"]
    assert out["scores"] == ref["scores"]
