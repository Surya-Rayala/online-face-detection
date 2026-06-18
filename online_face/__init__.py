"""online-face-detection — streaming, frame-by-frame face detection.

Lazy public API (importing ``online_face`` does not import torch):

    from online_face import FaceDetector
    det = FaceDetector("retinaface", device="auto")     # tensor/ndarray in
    res = det(frame)                                     # -> FaceFrameResult
"""
from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.3.1"

__all__ = ["FaceDetector", "FaceFrameResult", "FaceClient", "FaceStream",
           "available_models", "available_weights", "__version__"]


def __getattr__(name: str):
    if name in ("FaceDetector", "FaceFrameResult"):
        from .detector import FaceDetector, FaceFrameResult

        return {"FaceDetector": FaceDetector, "FaceFrameResult": FaceFrameResult}[name]
    if name == "FaceClient":
        from .client import FaceClient

        return FaceClient
    if name == "FaceStream":
        from .aio import FaceStream

        return FaceStream
    if name == "available_models":
        from .families import available_models

        return available_models
    if name == "available_weights":
        from .registry import available_weights

        return available_weights
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from .detector import FaceDetector, FaceFrameResult
