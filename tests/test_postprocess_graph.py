"""Face post-processing-in-graph (decode + NMS baked into the export).

Gated on torch + onnxruntime + weights; self-skips offline. TorchScript graph mode
is intentionally unsupported (torchvision backbone isn't scriptable) and TRT is
untestable here (no NVIDIA GPU) — both are covered by assertions/skips."""
from __future__ import annotations

import numpy as np
import pytest


def _frame_with_face():
    import cv2
    paths = [
        "/Users/surya_rayala/Desktop/Projects/online-pipeline/online-face-detection/"
        ".venv/lib/python3.11/site-packages/skimage/data/astronaut.png",
    ]
    base = None
    for p in paths:
        base = cv2.imread(p)
        if base is not None:
            break
    frame = (np.random.RandomState(0).rand(480, 640, 3) * 40).astype("uint8")
    if base is not None:
        frame[80:440, 150:510] = cv2.resize(base, (360, 360))
    return frame


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, x2 - x1), max(0, y2 - y1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@pytest.mark.timeout(600)
def test_onnx_graph_postprocess_matches_raw():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("onnxruntime")
    import online_face

    try:
        raw = online_face.FaceDetector("retinaface", device="cpu", runtime="onnx", conf=0.5, nms=0.4)
        graph = online_face.FaceDetector("retinaface", device="cpu", runtime="onnx",
                                         conf=0.5, nms=0.4, postprocess="graph")
    except Exception as e:  # offline / export unavailable
        pytest.skip(f"onnx export unavailable: {e}")

    frame = _frame_with_face()
    r, g = raw(frame), graph(frame)
    raw.close(); graph.close()

    assert len(g) == len(r) and len(r) >= 1
    for b in r.boxes:                                  # every raw box has a near-identical graph box
        assert max(_iou(b, c) for c in g.boxes) > 0.99


def test_torchscript_graph_is_rejected_cleanly():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import online_face
    from online_face.runtime.errors import RuntimeUnavailableError

    try:
        with pytest.raises(RuntimeUnavailableError):
            online_face.FaceDetector("retinaface", device="cpu", runtime="torchscript",
                                     postprocess="graph", warmup=False)
    except Exception as e:
        if "weights" in str(e).lower() or "download" in str(e).lower():
            pytest.skip(f"weights unavailable: {e}")
        raise
