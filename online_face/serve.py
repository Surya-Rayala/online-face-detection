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
from typing import Any, Dict, Optional, Sequence

from . import __version__

_INPUTS = [{"name": "frame", "type": "image", "required": True}]
_OUTPUTS = [
    {"name": "boxes", "type": "ndarray"},
    {"name": "scores", "type": "ndarray"},
    {"name": "landmarks", "type": "ndarray"},
    {"name": "shape", "type": "json"},
]


def create_app(model: str = "retinaface", *, weights=None, runtime: str = "auto",
               device: str = "auto", precision: str = "auto", conf: float = 0.5,
               nms: float = 0.4, input_size=None, stream_queue: int = 32,
               postprocess: str = "raw", max_faces: int = 256, inject_latency_ms: float = 0.0):
    """Build a FastAPI app wrapping one eagerly-constructed ``FaceDetector``."""
    import asyncio
    import time

    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse, Response
    from starlette.concurrency import run_in_threadpool

    from . import _wire
    from .detector import FaceDetector

    det = FaceDetector(model, weights=weights, runtime=runtime, device=device,
                       precision=precision, conf=conf, nms=nms, input_size=input_size,
                       warmup=True, postprocess=postprocess, max_faces=max_faces)

    app = FastAPI(title="online-face-detection", version=__version__)

    def _meta() -> Dict[str, Any]:
        cfg = det.config
        return {"name": "online_face", "modality": "vision", "model": model,
                "runtime": cfg.runtime, "device": cfg.device,
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
        res = det(inputs["frame"])
        if inject_latency_ms:                          # simulate network RTT (blocks this request)
            await asyncio.sleep(inject_latency_ms / 1000.0)
        if _wire.CT_NPZ in request.headers.get("accept", ""):
            # binary response: skip JSON .tolist() boxing (helps when many boxes)
            body = _wire.encode_npz(boxes=res.boxes, scores=res.scores,
                                    landmarks=res.landmarks, shape=res.shape)
            return Response(content=body, media_type=_wire.CT_NPZ)
        return {"outputs": _outputs(res), "stats": det.stats.as_dict()}

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

        async def _send(reply):
            async with send_lock:
                await ws.send_json(reply)

        async def receiver():
            while True:
                data = await ws.receive_bytes()                 # 4-byte id + encoded frame
                await q.put((data, time.perf_counter()))        # awaits -> socket backpressure

        async def worker():
            while True:
                data, t_recv = await q.get()
                try:
                    fid = int.from_bytes(data[:4], "big")
                    img = _wire.decode_image(data[4:])
                    t0 = time.perf_counter()
                    res = await run_in_threadpool(det, img)
                    infer_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                    reply = {"id": fid, "outputs": _outputs(res),
                             "server": {"infer_ms": infer_ms, "queue_depth": q.qsize(),
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

        tasks = [asyncio.ensure_future(receiver()), asyncio.ensure_future(worker())]
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
                     inject_latency_ms=args.inject_latency_ms)
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
