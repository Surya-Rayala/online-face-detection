"""Public annotate() helper (torch-free, numpy+opencv)."""
from __future__ import annotations

import numpy as np

import online_face
from online_face.client import FaceResult


def test_annotate_returns_new_image_with_drawing():
    frame = np.zeros((90, 120, 3), "uint8")
    res = FaceResult(np.array([[10, 10, 70, 70]], "float32"), np.array([0.88], "float32"),
                     np.zeros((1, 5, 2), "float32"), (90, 120))
    out = online_face.annotate(frame, res, hud="frame 0")
    assert out.shape == frame.shape
    assert out is not frame and not np.shares_memory(out, frame)
    assert out.any()                                  # something was drawn
    assert not frame.any()                            # original untouched


def test_annotate_handles_no_faces():
    frame = np.zeros((20, 20, 3), "uint8")
    res = FaceResult(np.zeros((0, 4), "float32"), np.zeros((0,), "float32"),
                     np.zeros((0, 5, 2), "float32"), (20, 20))
    out = online_face.annotate(frame, res)
    assert out.shape == frame.shape
