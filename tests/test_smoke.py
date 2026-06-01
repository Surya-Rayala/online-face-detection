"""Smoke tests. The network-dependent end-to-end test self-skips offline."""
from __future__ import annotations

import numpy as np
import pytest


def test_imports_and_discovery():
    import online_face

    assert online_face.__version__
    assert "retinaface" in online_face.available_models()
    weights = online_face.available_weights("retinaface")
    assert "mobilenet0.25" in weights and "ternaus_resnet50" in weights


def test_device_and_runtime_resolution():
    from online_face.runtime.device import resolve_device
    from online_face.runtime.backends import resolve_runtime

    dev = resolve_device("auto")
    assert dev.split(":")[0] in ("cuda", "mps", "cpu")
    assert resolve_runtime(dev, "torch") == "torch"
    assert resolve_runtime("cpu", "onnx") == "onnx"


def test_letterbox_and_rescale_roundtrip():
    torch = pytest.importorskip("torch")
    from online_face.runtime.tensor import letterbox, rescale_boxes

    img = torch.zeros((3, 480, 640))
    out, meta = letterbox(img, (640, 640), pad_value=0.0)
    assert out.shape == (3, 640, 640)
    # a box covering the resized content maps back to ~full original frame
    full = torch.tensor([[meta.pad_x, meta.pad_y, meta.pad_x + 640 * meta.scale, meta.pad_y + 480 * meta.scale]])
    back = rescale_boxes(full, meta)
    assert abs(float(back[0, 2]) - 639) < 2 and abs(float(back[0, 3]) - 479) < 2


def test_priors_order_and_count():
    pytest.importorskip("torch")
    from online_face.models.retinaface import prior_box

    priors = prior_box((640, 640), "cpu")
    # 3 feature maps at strides 8/16/32, 2 anchors each: 80*80*2 + 40*40*2 + 20*20*2 = 16800
    assert priors.shape == (16800, 4)


@pytest.mark.timeout(300)
def test_end_to_end_detect_on_synthetic_frame():
    """Builds the default detector (downloads ~1.7MB weights) and runs one frame."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import online_face

    try:
        det = online_face.FaceDetector("retinaface", device="auto", warmup=False)
    except Exception as e:  # offline / download failure
        pytest.skip(f"weights unavailable: {e}")
    frame = (np.random.rand(360, 640, 3) * 255).astype("uint8")
    res = det(frame)
    assert res.boxes.shape[1] == 4
    assert res.landmarks.shape[1:] == (5, 2)
    assert len(res) == res.scores.shape[0]
    det.close()
