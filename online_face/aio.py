"""Long-lived async streaming session (install with the ``[client]`` extra).

One ``FaceStream`` holds WebSocket connections across a **pool of endpoints**, lets
you **push** frames (each with arbitrary metadata) without buffering, and yields
results **as they complete** (out of order) tagged with your metadata. An adaptive
controller (see :mod:`._autoscale`) ramps in-flight concurrency and the number of
connections toward a target fps/latency, scaling out while the bottleneck is the
network and holding when the model saturates.

Wire (stream protocol v2.1): one **binary** message per frame — a 4-byte big-endian
id followed by the encoded image; the reply is one JSON
``{"id", "outputs", "server": {...}}``. One message per direction keeps the hot
path cheap. Torch-free: ``asyncio`` + ``websockets`` + numpy/opencv via ``_wire``.

    async with FaceStream(["http://gpu0:8001", "http://gpu1:8001"],
                          target_fps=30, target_latency_ms=150, max_side=960) as s:
        async def pump():
            for frame, info in source():
                await s.push(frame, meta=info)
            await s.aclose()
        asyncio.create_task(pump())
        async for result, info in s.results():   # out of order
            handle(info, result)
"""
from __future__ import annotations

import asyncio
import json
import struct
import time
from typing import Any, AsyncIterator, Optional, Tuple

import numpy as np

from ._autoscale import AutoScaler
from ._wire import downscale_to_maxside, encode_image
from .client import FaceResult, build_face_result

_SENTINEL = object()


def _ws_url(url: str) -> str:
    u = url.rstrip("/").replace("https://", "wss://").replace("http://", "ws://")
    return u + "/stream"


class _Gate:
    """A cheap, resizable concurrency limiter (lighter than a Condition on the hot
    path). ``set_limit`` grows by releasing permits; shrinking is absorbed lazily
    on the next releases (good enough for the autoscaler)."""

    def __init__(self) -> None:
        self._sem = asyncio.Semaphore(0)
        self._limit = 0
        self._debt = 0
        self.inflight = 0

    def set_limit(self, n: int) -> None:
        n = max(0, int(n))
        d = n - self._limit
        self._limit = n
        if d > 0:
            pay = min(self._debt, d)
            self._debt -= pay
            for _ in range(d - pay):
                self._sem.release()
        elif d < 0:
            self._debt += -d

    async def acquire(self) -> None:
        await self._sem.acquire()
        self.inflight += 1

    def release(self) -> None:
        self.inflight -= 1
        if self._debt > 0:
            self._debt -= 1
        else:
            self._sem.release()

    def unblock(self, k: int) -> None:
        """Release ``k`` permits to wake senders parked on acquire (used at close)."""
        self._debt = 0
        for _ in range(max(0, int(k))):
            self._sem.release()


class FaceStream:
    def __init__(self, urls, *, target_fps: Optional[float] = None,
                 target_latency_ms: Optional[float] = None,
                 encode: str = "jpeg", quality: int = 90, max_side: Optional[int] = None,
                 min_inflight: int = 1, max_inflight: int = 64, max_queue: int = 256,
                 max_connections: Optional[int] = None, open_timeout: float = 30.0,
                 tick_s: float = 0.5, connect=None) -> None:
        self._urls = [urls] if isinstance(urls, str) else list(urls)
        if not self._urls:
            raise ValueError("FaceStream needs at least one URL")
        self.encode = encode
        self.quality = quality
        self.max_side = max_side
        self.open_timeout = open_timeout
        self.tick_s = tick_s
        self._connect = connect
        self._scaler = AutoScaler(
            target_fps=target_fps, target_latency_ms=target_latency_ms,
            min_inflight=min_inflight, max_inflight=max_inflight,
            max_connections=max_connections or max(1, len(self._urls)))
        self._min_inflight = max(1, int(min_inflight))
        self._max_queue = max_queue
        self._in_q: Optional[asyncio.Queue] = None
        self._out_q: Optional[asyncio.Queue] = None
        self._gate: Optional[_Gate] = None
        self._conns: list = []
        self._ctrl_task = None
        self._pending: dict = {}
        self._next_id = 0
        self._started = False
        self._closing = False

    # -- lifecycle ---------------------------------------------------------
    async def _start(self) -> None:
        if self._started:
            return
        self._in_q = asyncio.Queue(maxsize=self._max_queue)
        self._out_q = asyncio.Queue()
        self._gate = _Gate()
        self._gate.set_limit(self._min_inflight)
        await self._open_conn(self._urls[0])
        self._ctrl_task = asyncio.ensure_future(self._controller())
        self._started = True

    async def __aenter__(self) -> "FaceStream":
        await self._start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def aclose(self, drain_timeout: float = 10.0) -> None:
        if not self._started or self._closing:
            return
        deadline = time.perf_counter() + drain_timeout
        while self._in_q is not None and not self._in_q.empty() and time.perf_counter() < deadline:
            await asyncio.sleep(0.01)
        # drain in-flight replies, but bail if it stops making progress (~1s) so a
        # lost/straggler reply can't stall close for the whole timeout
        last, stall = len(self._pending), 0
        while self._pending and time.perf_counter() < deadline:
            await asyncio.sleep(0.05)
            if len(self._pending) == last:
                stall += 1
            else:
                last, stall = len(self._pending), 0
            if stall >= 20:
                break
        self._closing = True
        self._gate.unblock(len(self._conns) + 1)      # wake any sender parked on acquire
        for _ in self._conns:
            await self._in_q.put(None)
        if self._ctrl_task:
            self._ctrl_task.cancel()
        for c in list(self._conns):
            await self._close_conn(c)
        await self._out_q.put(_SENTINEL)

    # -- public API --------------------------------------------------------
    async def push(self, frame: np.ndarray, meta: Any = None) -> None:
        """Enqueue a frame (+ opaque metadata). Awaits only when the internal
        queue is full (backpressure)."""
        if self._closing:
            raise RuntimeError("FaceStream is closing")
        await self._start()
        frame = np.asarray(frame)
        oh, ow = frame.shape[:2]
        sent, scale = downscale_to_maxside(frame, self.max_side)
        data, _ = encode_image(sent, self.encode, self.quality)
        fid = self._next_id
        self._next_id += 1
        self._pending[fid] = {"h": oh, "w": ow, "scale": scale, "meta": meta, "t_send": None}
        await self._in_q.put((fid, struct.pack(">I", fid) + data))   # one binary message

    async def results(self) -> AsyncIterator[Tuple[FaceResult, Any]]:
        """Yield ``(FaceResult, meta)`` as each frame completes — out of order."""
        await self._start()
        while True:
            item = await self._out_q.get()
            if item is _SENTINEL:
                return
            yield item

    def stats(self) -> dict:
        sc = self._scaler
        return {"conns": len(self._conns), "target_inflight": sc.target_inflight,
                "inflight": (self._gate.inflight if self._gate else 0),
                "queue": (self._in_q.qsize() if self._in_q else 0),
                "bound": sc.bound, "rtt_ms": round(sc._rtt.get(), 2),
                "srv_ms": round(sc._srv.get(), 2), "infer_ms": round(sc._infer.get(), 2),
                "queue_depth": round(sc._qd.get(), 2)}

    # -- connections -------------------------------------------------------
    async def _open_conn(self, url: str) -> None:
        ws = await self._dial(_ws_url(url))
        conn = {"ws": ws, "url": url, "tasks": []}
        conn["tasks"] = [asyncio.ensure_future(self._sender(conn)),
                         asyncio.ensure_future(self._receiver(conn))]
        self._conns.append(conn)

    async def _dial(self, ws_url: str):
        if self._connect is not None:
            return await self._connect(ws_url)
        import websockets
        return await websockets.connect(ws_url, open_timeout=self.open_timeout,
                                        max_size=None, compression=None)

    async def _close_conn(self, conn: dict) -> None:
        if conn in self._conns:
            self._conns.remove(conn)
        for t in conn["tasks"]:
            t.cancel()
        try:
            await conn["ws"].close()
        except Exception:
            pass

    async def _sender(self, conn: dict) -> None:
        ws = conn["ws"]
        try:
            while not self._closing:
                item = await self._in_q.get()
                if item is None:
                    break
                await self._gate.acquire()
                if self._closing:
                    break
                fid, payload = item
                if fid in self._pending:
                    self._pending[fid]["t_send"] = time.perf_counter()
                await ws.send(payload)
        except asyncio.CancelledError:
            pass
        except Exception:
            return

    async def _receiver(self, conn: dict) -> None:
        ws = conn["ws"]
        try:
            while not self._closing:
                msg = await ws.recv()
                now = time.perf_counter()
                rep = json.loads(msg)
                info = self._pending.pop(rep.get("id"), None)
                self._gate.release()
                if info is None:
                    continue
                result = build_face_result(rep["outputs"], info["h"], info["w"], info["scale"])
                srv = rep.get("server", {})
                t_send = info.get("t_send") or now
                self._scaler.observe(
                    rtt_ms=(now - t_send) * 1000.0,
                    srv_ms=(srv.get("t_send", 0.0) - srv.get("t_recv", 0.0)) * 1000.0,
                    infer_ms=srv.get("infer_ms", 0.0), queue_depth=srv.get("queue_depth", 0.0))
                await self._out_q.put((result, info["meta"]))
        except asyncio.CancelledError:
            pass
        except Exception:
            return

    async def _controller(self) -> None:
        try:
            while not self._closing:
                await asyncio.sleep(self.tick_s)
                st = self._scaler.tick(self.tick_s)
                self._gate.set_limit(st.target_inflight)
                while len(self._conns) < st.n_conn:
                    await self._open_conn(self._urls[len(self._conns) % len(self._urls)])
                while len(self._conns) > st.n_conn and len(self._conns) > 1:
                    await self._close_conn(self._conns[-1])
        except asyncio.CancelledError:
            pass
