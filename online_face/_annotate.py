"""Draw detections onto a frame (install with the ``[client]`` extra).

Torch-free public helper (numpy + opencv via :mod:`.runtime.viz`) that works on
either a :class:`~online_face.FaceFrameResult` (in-process) or a
:class:`~online_face.client.FaceResult` (streaming/remote mirror) — both expose
``boxes`` / ``scores`` / ``landmarks``. Returns a new annotated image; the input is
not modified.

    from online_face import annotate
    async for result, meta in stream.results():
        cv2.imshow("faces", annotate(meta["frame"], result, hud=f"frame {meta['i']}"))
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def annotate(frame: np.ndarray, result, *, draw_landmarks: bool = True,
             show_scores: bool = True, thickness: int = 2,
             hud: Optional[str] = None) -> np.ndarray:
    """Return a copy of ``frame`` with each face box (+ score / landmarks) drawn.

    ``result`` may be a ``FaceFrameResult`` / ``FaceResult`` or any object exposing
    ``boxes`` (N,4 xyxy), optional ``scores`` (N,), optional ``landmarks`` (N,5,2),
    all in the coordinate space of ``frame``. ``hud`` draws a top-left status line.
    """
    from .runtime.viz import draw_box_label, draw_hud, draw_points, hash_color

    img = np.ascontiguousarray(np.asarray(frame).copy())
    boxes = getattr(result, "boxes", result)
    scores = getattr(result, "scores", None)
    landmarks = getattr(result, "landmarks", None)
    boxes = np.asarray(boxes, dtype="float32").reshape(-1, 4)
    for i, box in enumerate(boxes):
        label = None
        if show_scores and scores is not None and i < len(scores):
            label = f"{float(scores[i]):.2f}"
        draw_box_label(img, box, hash_color(i), label, thickness=thickness)
    if draw_landmarks and landmarks is not None:
        for lm in np.asarray(landmarks, dtype="float32").reshape(-1, 5, 2):
            draw_points(img, lm, (0, 255, 255), max(1, thickness))
    if hud:
        draw_hud(img, hud)
    return img
