"""Optional HTTP serving adapter (install with the ``[serve]`` extra).

Exposes ``FaceDetector`` behind the project's uniform, modality-generalizable
contract so the model can be hosted as its own service (local process or cloud
instance) and composed by URL:

  * ``GET  /meta``    self-describing: name, modality, named typed inputs/outputs
  * ``GET  /healthz`` readiness + device/runtime
  * ``POST /predict`` multipart: one typed part per named input -> JSON named outputs

``fastapi``/``uvicorn`` are imported lazily inside functions, so importing
``online_face`` never requires the serving deps. The detector is built eagerly in
:func:`create_app` (before uvicorn binds the port), so any first-run artifact
export finishes before the service accepts traffic.

NOTE: this module deliberately does NOT use ``from __future__ import annotations``.
FastAPI resolves endpoint type hints via ``get_type_hints``; with stringized
annotations it cannot resolve ``Request`` (imported inside ``create_app`` to keep
fastapi optional) and would mistake the param for a query field.
"""
import argparse
import asyncio
from typing import Any, Dict, List, Optional, Sequence, Union

from . import __version__

_INPUTS = [{"name": "frame", "type": "image", "required": True}]
_OUTPUTS = [
    {"name": "boxes", "type": "ndarray"},
    {"name": "scores", "type": "ndarray"},
    {"name": "landmarks", "type": "ndarray"},
    {"name": "shape", "type": "json"},
]


def _parse_instances(spec) -> Union[int, Dict[int, int]]:
    """Parse an ``--instances`` value into a plain int N or a ``{gpu_index: count}`` map.

    Accepts an int (``4``), a bare number string (``"4"``), a dict (``{0: 2, 1: 1}``),
    or a per-GPU spec string (``"0=2,1=1"`` / ``"cuda:0=2,cuda:1=1"``). Raises
    ``ValueError`` on anything malformed so the caller can warn and fall back."""
    if isinstance(spec, bool):                          # guard: bool is an int subclass
        raise ValueError("bool")
    if isinstance(spec, int):
        return spec
    if isinstance(spec, dict):
        return {int(k): int(v) for k, v in spec.items()}
    s = str(spec).strip()
    if not s:
        raise ValueError("empty")
    if "=" not in s:
        return int(s)                                   # bare number (may raise ValueError)
    out: Dict[int, int] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        key, _, val = part.partition("=")
        key = key.strip().lower().replace("cuda:", "")
        out[int(key)] = int(val)
    if not out:
        raise ValueError("no entries")
    return out


def _resolve_instance_devices(spec, device) -> List[str]:
    """Resolve ``--instances`` into a per-instance torch device list.

    Warns and degrades gracefully on bad input: an unparseable spec or a non-CUDA
    box asked for per-GPU placement falls back sensibly, and missing GPU indices
    are skipped. Always returns at least one device."""
    from .runtime.device import resolve_device
    from .runtime.logging import get_logger

    log = get_logger("serve")
    resolved = resolve_device(device)
    try:
        parsed = _parse_instances(spec)
    except (ValueError, TypeError):
        log.warning("could not parse --instances %r; using 1", spec)
        return [resolved]
    try:
        import torch

        cuda_count = int(torch.cuda.device_count())
    except Exception:
        cuda_count = 0

    if isinstance(parsed, int):                         # plain N, auto-placed
        if parsed <= 0:
            log.warning("--instances=%r is not positive; using 1", spec)
            return [resolved]
        if resolved.startswith("cuda") and cuda_count > 1:
            return [f"cuda:{i % cuda_count}" for i in range(parsed)]   # round-robin GPUs
        if parsed > 1:
            log.warning("%d instances share %s (they time-share compute); multi-instance "
                        "mainly helps across multiple GPUs", parsed, resolved)
        return [resolved] * parsed

    if cuda_count == 0:                                 # per-GPU map but no CUDA
        total = sum(c for c in parsed.values() if c > 0) or 1
        log.warning("per-GPU --instances %r needs CUDA (device=%s); placing all %d on %s",
                    spec, resolved, total, resolved)
        return [resolved] * total

    devices: List[str] = []
    for idx in sorted(parsed):
        count = parsed[idx]
        if count <= 0:
            continue
        if idx >= cuda_count:
            log.warning("cuda:%d not found (only %d GPU(s) present); skipping its %d instance(s)",
                        idx, cuda_count, count)
            continue
        devices.extend([f"cuda:{idx}"] * count)
    if not devices:
        log.warning("--instances=%r selected no valid GPUs; using 1 on %s", spec, resolved)
        return [resolved]
    return devices


class _Pool:
    """Fixed-size checkout pool of model instances.

    ``run`` borrows a free instance, runs ``fn(instance)`` in a worker thread (so the
    event loop keeps serving and concurrent requests overlap decode/transfer with
    compute), and returns it. One instance == today's behavior plus a threadpool hop;
    N>1 lets N requests run at once (true parallelism only across distinct GPUs). The
    count is fixed by the operator via ``--instances``; there is no auto-regulation."""

    def __init__(self, instances) -> None:
        self.instances = list(instances)
        self._free: asyncio.Queue = asyncio.Queue()
        for inst in self.instances:
            self._free.put_nowait(inst)

    def __len__(self) -> int:
        return len(self.instances)

    async def run(self, fn):
        from starlette.concurrency import run_in_threadpool

        inst = await self._free.get()
        try:
            return await run_in_threadpool(fn, inst)
        finally:
            self._free.put_nowait(inst)


def create_app(model: str = "retinaface", *, weights=None, runtime: str = "auto",
               device: str = "auto", precision: str = "auto", conf: float = 0.5,
               nms: float = 0.4, input_size=None, stream_queue: int = 32,
               postprocess: str = "raw", max_faces: int = 256, inject_latency_ms: float = 0.0,
               instances: Union[int, str, dict] = 1):
    """Build a FastAPI app wrapping a pool of ``instances`` ``FaceDetector``s.

    ``instances`` defaults to 1 (one model, today's behavior). It accepts an int,
    a ``"0=2,1=1"`` per-GPU spec string, or a ``{gpu_index: count}`` dict; the pool
    pins one model per GPU on a multi-GPU box (else N copies share one device) and
    dispatches requests across them through a checkout pool."""
    import time

    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse, Response
    from starlette.concurrency import run_in_threadpool

    from . import _wire
    from .detector import FaceDetector

    inst_devices = _resolve_instance_devices(instances, device)
    dets = [FaceDetector(model, weights=weights, runtime=runtime, device=d,
                         precision=precision, conf=conf, nms=nms, input_size=input_size,
                         warmup=True, postprocess=postprocess, max_faces=max_faces)
            for d in inst_devices]
    det = dets[0]                                  # config/device identical across instances
    pool = _Pool(dets)

    app = FastAPI(title="online-face-detection", version=__version__)

    def _meta() -> Dict[str, Any]:
        cfg = det.config
        return {"name": "online_face", "modality": "vision", "model": model,
                "runtime": cfg.runtime, "device": cfg.device,
                "instances": len(pool), "instance_devices": inst_devices,
                "inputs": _INPUTS, "outputs": _OUTPUTS, "stream_protocol": 2}

    def _outputs(res) -> Dict[str, Any]:
        return {"boxes": res.boxes.tolist(), "scores": res.scores.tolist(),
                "landmarks": res.landmarks.tolist(), "shape": list(res.shape)}

    @app.get("/meta")
    def meta() -> Dict[str, Any]:
        return _meta()

    @app.get("/healthz")
    def healthz() -> Dict[str, Any]:
        m = _meta()
        m["ready"] = True
        m["mps"] = det.device.startswith("mps")
        return m

    @app.post("/predict")
    async def predict(request: Request):
        form = await request.form()
        inputs: Dict[str, Any] = {}
        for key, val in form.items():
            if hasattr(val, "read"):                       # UploadFile (typed part)
                inputs[key] = _wire.decode_part(getattr(val, "content_type", None), await val.read())
            else:                                          # plain text field -> JSON/scalar
                inputs[key] = _wire.decode_part(_wire.CT_JSON, str(val).encode("utf-8"))
        if "frame" not in inputs:
            return JSONResponse({"error": "missing required input 'frame'"}, status_code=422)
        frame = inputs["frame"]
        # run on a pooled instance in a worker thread, so concurrent requests overlap
        # decode/transfer with compute (and, across instances/GPUs, run in parallel)
        res, stats = await pool.run(lambda d: (d(frame), d.stats.as_dict()))
        if inject_latency_ms:                          # simulate network RTT (blocks this request)
            await asyncio.sleep(inject_latency_ms / 1000.0)
        if _wire.CT_NPZ in request.headers.get("accept", ""):
            # binary response: skip JSON .tolist() boxing (helps when many boxes)
            body = _wire.encode_npz(boxes=res.boxes, scores=res.scores,
                                    landmarks=res.landmarks, shape=res.shape)
            return Response(content=body, media_type=_wire.CT_NPZ)
        return {"outputs": _outputs(res), "stats": stats}

    @app.websocket("/stream")
    async def stream(ws: WebSocket):
        """Stream protocol v2.1 — pipelined, id-tagged, with server telemetry.

        Per frame the client sends ONE binary message: a 4-byte big-endian id
        followed by the encoded image. The reply is one JSON ``{"id", "outputs",
        "server": {infer_ms, queue_depth, t_recv, t_send}}``. A receive loop drains
        into a bounded queue (backpressure) while a worker runs inference in a
        threadpool, so the client can keep many frames in flight and use the
        telemetry to autoscale. Replies carry the id (out-of-order safe)."""
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=max(1, int(stream_queue)))
        send_lock = asyncio.Lock()
        n_instances = len(pool)

        async def _send(reply):
            async with send_lock:
                await ws.send_json(reply)

        async def receiver():
            while True:
                data = await ws.receive_bytes()                 # 4-byte id + encoded frame
                await q.put((data, time.perf_counter()))        # awaits -> socket backpressure

        async def worker(det_i):                                # one task per pooled instance
            while True:
                data, t_recv = await q.get()
                try:
                    fid = int.from_bytes(data[:4], "big")
                    img = _wire.decode_image(data[4:])
                    t0 = time.perf_counter()
                    res = await run_in_threadpool(det_i, img)
                    infer_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                    reply = {"id": fid, "outputs": _outputs(res),
                             "server": {"infer_ms": infer_ms, "queue_depth": q.qsize(),
                                        "instances": n_instances,
                                        "t_recv": t_recv, "t_send": time.perf_counter()}}
                    if inject_latency_ms:                       # RTT is overlap-able here (pipelining)
                        asyncio.ensure_future(_delayed(reply))
                    else:
                        await _send(reply)
                finally:
                    q.task_done()

        async def _delayed(reply):
            await asyncio.sleep(inject_latency_ms / 1000.0)
            await _send(reply)

        tasks = [asyncio.ensure_future(receiver())]
        tasks += [asyncio.ensure_future(worker(d)) for d in pool.instances]
        try:
            await asyncio.gather(*tasks)
        except WebSocketDisconnect:
            pass
        finally:
            for t in tasks:
                t.cancel()

    return app


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser("online-face-serve",
                                description="Serve RetinaFace face detection over HTTP (uniform contract).")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--model", default="retinaface")
    p.add_argument("--weights", default=None)
    p.add_argument("--runtime", default="auto", choices=["auto", "torch", "torchscript", "onnx", "trt"])
    p.add_argument("--device", default="auto")
    p.add_argument("--instances", default="1",
                   help="server-side inference instances: a number ('4') or per-GPU map "
                        "('cuda:0=2,cuda:1=1'). On one device N copies just time-share it (N x memory); "
                        "running more than one instance mainly helps across multiple GPUs")
    p.add_argument("--precision", default="auto", choices=["auto", "fp32", "fp16", "int8"])
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--nms", type=float, default=0.4)
    p.add_argument("--input-size", type=int, default=None)
    p.add_argument("--stream-queue", type=int, default=32,
                   help="max frames buffered per /stream connection (backpressure bound)")
    p.add_argument("--postprocess", default="raw", choices=["raw", "graph"],
                   help="'graph' bakes decode+NMS into the exported engine (onnx/trt); conf/nms become fixed")
    p.add_argument("--max-faces", type=int, default=256, help="padded detection cap for postprocess='graph'")
    p.add_argument("--inject-latency-ms", type=float, default=0.0,
                   help="simulate network RTT per request (debug/benchmark): blocks /predict, overlap-able on /stream")
    args = p.parse_args(argv)

    import uvicorn

    app = create_app(args.model, weights=args.weights, runtime=args.runtime, device=args.device,
                     precision=args.precision, conf=args.conf, nms=args.nms, input_size=args.input_size,
                     stream_queue=args.stream_queue, postprocess=args.postprocess, max_faces=args.max_faces,
                     inject_latency_ms=args.inject_latency_ms, instances=args.instances)
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
