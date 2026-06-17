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

from ._wire import CT_NPZ, decode_npz, downscale_to_maxside, encode_image


@dataclass(frozen=True)
class FaceResult:
    boxes: np.ndarray          # (N, 4) xyxy
    scores: np.ndarray         # (N,)
    landmarks: np.ndarray      # (N, 5, 2)
    shape: Tuple[int, ...]     # (H, W)

    def __len__(self) -> int:
        return int(self.boxes.shape[0])


def build_face_result(out: dict, orig_h: int, orig_w: int, scale: float) -> "FaceResult":
    """Build a FaceResult from server ``outputs``, rescaling sent-frame coords back
    UP to original-frame coords when the client downscaled (``scale != 1.0``).
    Shared by FaceClient.predict and the streaming client."""
    boxes = np.asarray(out["boxes"], dtype="float32").reshape(-1, 4)
    landmarks = np.asarray(out["landmarks"], dtype="float32").reshape(-1, 5, 2)
    scores = np.asarray(out["scores"], dtype="float32").reshape(-1)
    if scale != 1.0:
        boxes /= scale
        landmarks /= scale
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, orig_w - 1)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, orig_h - 1)
        landmarks[..., 0] = landmarks[..., 0].clip(0, orig_w - 1)
        landmarks[..., 1] = landmarks[..., 1].clip(0, orig_h - 1)
    return FaceResult(boxes, scores, landmarks, (orig_h, orig_w))


def _pipeline(fn, items, max_workers: int):
    """Run ``fn`` over ``items`` with up to ``max_workers`` calls in flight, yielding
    results in input order. Generic over the per-item call (plain HTTP overlap)."""
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    ex = ThreadPoolExecutor(max_workers=max(1, int(max_workers)))
    try:
        it = iter(items)
        window: deque = deque()
        for _ in range(max(1, int(max_workers))):
            try:
                window.append(ex.submit(fn, next(it)))
            except StopIteration:
                break
        while window:
            result = window.popleft().result()
            try:
                window.append(ex.submit(fn, next(it)))
            except StopIteration:
                pass
            yield result
    finally:
        ex.shutdown(wait=False)


class FaceClient:
    """Remote proxy mirroring ``FaceDetector``'s per-frame call surface."""

    def __init__(self, url: str = "http://127.0.0.1:8001", *, encode: str = "jpeg",
                 quality: int = 90, max_side: Optional[int] = None,
                 binary_response: bool = False,
                 timeout: float = 30.0, session: Optional[Any] = None) -> None:
        self.url = url.rstrip("/")
        self.encode = encode
        self.quality = quality
        self.max_side = max_side
        self.binary_response = binary_response
        self.timeout = timeout
        if session is None:
            import requests
            session = requests.Session()
        self._session = session

    def healthz(self) -> Dict[str, Any]:
        return self._session.get(f"{self.url}/healthz", timeout=self.timeout).json()

    def meta(self) -> Dict[str, Any]:
        return self._session.get(f"{self.url}/meta", timeout=self.timeout).json()

    def predict(self, frame: np.ndarray, *, frame_index: Optional[int] = None,
                max_side: Optional[int] = None) -> FaceResult:
        frame = np.asarray(frame)
        orig_h, orig_w = frame.shape[:2]
        ms = self.max_side if max_side is None else max_side
        sent, scale = downscale_to_maxside(frame, ms)
        data, ct = encode_image(sent, self.encode, self.quality)
        files = {"frame": (f"frame.{self.encode}", data, ct)}
        headers = {"Accept": CT_NPZ} if self.binary_response else None
        r = self._session.post(f"{self.url}/predict", files=files, timeout=self.timeout, headers=headers)
        r.raise_for_status()
        if CT_NPZ in r.headers.get("content-type", ""):
            out = decode_npz(r.content)            # arrays; build_face_result accepts them
        else:
            out = r.json()["outputs"]
        return build_face_result(out, orig_h, orig_w, scale)

    __call__ = predict

    def predict_stream(self, frames, *, max_workers: int = 4,
                       max_side: Optional[int] = None):
        """Overlap encode + network round-trip + parse across frames using a thread
        pool over the pooled keep-alive Session. Yields FaceResult in input order.
        Hides WAN latency without a persistent socket; ``max_workers`` ≈ how many
        frames to keep in flight. Torch-free (stdlib threads + requests)."""
        return _pipeline(lambda f: self.predict(f, max_side=max_side), frames, max_workers)

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "FaceClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
