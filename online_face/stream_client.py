"""Persistent WebSocket streaming client (install with the ``[client]`` extra).

Keeps a warm connection to an ``online-face-serve`` ``/stream`` endpoint and
pipelines up to ``max_inflight`` frames so network round-trip latency overlaps
encode + server inference — the right shape for LAN/remote deployments where RTT
dominates. Torch-free (``websockets`` + ``numpy`` + ``opencv``).

Replies are FIFO (the server runs one model sequentially), so frames are matched
to results by send order — no per-message id needed.
"""
from __future__ import annotations

import json
import queue
import threading
from typing import Any, Iterable, Iterator, Optional

import numpy as np

from ._wire import downscale_to_maxside, encode_image
from .client import FaceResult, build_face_result


class FaceStreamClient:
    def __init__(self, url: str = "http://127.0.0.1:8001", *, encode: str = "jpeg",
                 quality: int = 90, max_side: Optional[int] = None,
                 max_inflight: int = 4, open_timeout: float = 30.0) -> None:
        self.ws_url = (url.rstrip("/").replace("https://", "wss://").replace("http://", "ws://")
                       + "/stream")
        self.encode = encode
        self.quality = quality
        self.max_side = max_side
        self.max_inflight = max(1, int(max_inflight))
        self.open_timeout = open_timeout

    def predict_stream(self, frames: Iterable[np.ndarray]) -> Iterator[FaceResult]:
        """Yield a FaceResult per input frame, in input order, pipelined."""
        from websockets.sync.client import connect

        _DONE = object()
        with connect(self.ws_url, open_timeout=self.open_timeout, max_size=None) as ws:
            sem = threading.Semaphore(self.max_inflight)
            meta: "queue.Queue[Any]" = queue.Queue()
            err: list = []

            def sender():
                try:
                    for f in frames:
                        f = np.asarray(f)
                        oh, ow = f.shape[:2]
                        sent, scale = downscale_to_maxside(f, self.max_side)
                        sem.acquire()                      # bound in-flight frames
                        data, _ = encode_image(sent, self.encode, self.quality)
                        meta.put((oh, ow, scale))
                        ws.send(data)                      # binary frame
                except Exception as e:                     # surface to consumer
                    err.append(e)
                finally:
                    meta.put(_DONE)

            t = threading.Thread(target=sender, daemon=True)
            t.start()
            while True:
                item = meta.get()
                if item is _DONE:
                    break
                oh, ow, scale = item
                reply = ws.recv()                          # FIFO -> matches this frame
                sem.release()
                out = json.loads(reply)["outputs"]
                yield build_face_result(out, oh, ow, scale)
            t.join()
            if err:
                raise err[0]
