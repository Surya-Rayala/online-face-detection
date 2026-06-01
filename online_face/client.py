"""Lightweight HTTP client proxy (install with the ``[client]`` extra).

Torch-free (``requests`` + ``numpy`` + ``opencv`` only): talks to an
``online-face-serve`` endpoint with the same call shape as the local
``FaceDetector`` — ``client(frame) -> FaceResult`` — so a remote pipeline reads
exactly like an in-process one. Returns its own light result mirror (it never
imports ``detector``/the torch runtime).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ._wire import encode_image


@dataclass(frozen=True)
class FaceResult:
    boxes: np.ndarray          # (N, 4) xyxy
    scores: np.ndarray         # (N,)
    landmarks: np.ndarray      # (N, 5, 2)
    shape: Tuple[int, ...]     # (H, W)

    def __len__(self) -> int:
        return int(self.boxes.shape[0])


class FaceClient:
    """Remote proxy mirroring ``FaceDetector``'s per-frame call surface."""

    def __init__(self, url: str = "http://127.0.0.1:8001", *, encode: str = "png",
                 timeout: float = 30.0, session: Optional[Any] = None) -> None:
        import requests

        self.url = url.rstrip("/")
        self.encode = encode
        self.timeout = timeout
        self._session = session or requests.Session()

    def healthz(self) -> Dict[str, Any]:
        return self._session.get(f"{self.url}/healthz", timeout=self.timeout).json()

    def meta(self) -> Dict[str, Any]:
        return self._session.get(f"{self.url}/meta", timeout=self.timeout).json()

    def predict(self, frame: np.ndarray, *, frame_index: Optional[int] = None) -> FaceResult:
        data, ct = encode_image(np.asarray(frame), self.encode)
        files = {"frame": (f"frame.{self.encode}", data, ct)}
        r = self._session.post(f"{self.url}/predict", files=files, timeout=self.timeout)
        r.raise_for_status()
        out = r.json()["outputs"]
        return FaceResult(
            np.asarray(out["boxes"], dtype="float32").reshape(-1, 4),
            np.asarray(out["scores"], dtype="float32").reshape(-1),
            np.asarray(out["landmarks"], dtype="float32").reshape(-1, 5, 2),
            tuple(out.get("shape", ())),
        )

    __call__ = predict

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "FaceClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
