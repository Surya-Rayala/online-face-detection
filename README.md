# online-face-detection

Streaming, **frame-by-frame** face detection for real-time pipelines:
one small object — a frame in, structured results out. Runs under **torch / torchscript /
onnx / tensorrt** with export-once caching, on CPU, CUDA, Apple Silicon (MPS), and Jetson.

```python
from online_face import FaceDetector
det = FaceDetector("retinaface", device="auto")
res = det(frame)        # res.boxes, res.scores, res.landmarks
```

> **Models today:** RetinaFace. More face-detection families plug in via the registry — coming later.

---

## Install

Pick a runtime extra **and** an OpenCV build (`gui` for desktop, `headless` for servers/Docker/Jetson):

```bash
pip install "online-face-detection[torch,gui]"        # desktop (enables cv2 windows / --display)
pip install "online-face-detection[torch,headless]"   # servers, Docker, Jetson (no GTK/X11)
```

`[torch]` is the default runtime and works on CPU, CUDA, and Mac (MPS). **OpenCV is not bundled by
default** — you choose `[gui]` (`opencv-python`) or `[headless]` (`opencv-python-headless`). On
**Jetson** the GUI build pulls X11/GTK and frequently fails to build against the CUDA toolchain
(`nvcc` errors), so use `[headless]` — or install with **neither** and rely on JetPack's system
OpenCV (also use JetPack's torch/onnxruntime there, not the PyPI wheels). Other backends (`onnx`,
`tensorrt`, serving, client) are optional extras — see [Install options](#install-options).
(Prefer `uv`? See [Misc](#misc).)

---

## Use it (Python)

```python
from online_face import FaceDetector

det = FaceDetector(
    "retinaface",       # model family (the only one today)
    device="auto",      # "auto" (CUDA > MPS > CPU) | "cpu" | "cuda" | "mps"
    runtime="auto",     # "auto" | "torch" | "torchscript" | "onnx" | "trt"
    conf=0.5,           # detection confidence threshold
    nms=0.4,            # NMS IoU threshold
)
res = det(frame)        # weights auto-download on first use; see "Input" below

res.boxes        # (N, 4) xyxy, in original-frame coordinates
res.scores       # (N,)
res.landmarks    # (N, 5, 2)
```

**Input — what `frame` must be:** a NumPy array in **BGR** order, shape **`(H, W, 3)`**, dtype
**`uint8`** (OpenCV's native format — e.g. straight from `cv2.imread(...)` or
`cv2.VideoCapture(...).read()`), **or** a `torch.Tensor` of shape **`(3, H, W)`**. Any
resolution; the model letterboxes internally.

Drive a video file or a live stream (FPS/latency print to the terminal):

```python
for frame_ref, res in det.run_source("video.mp4"):                          # a file
    ...
for frame_ref, res in det.run_source("rtsp://cam/stream", is_stream=True):  # live
    ...
for frame_ref, res in det.run_source("video.mp4", is_stream=True):          # file as a stream
    ...
# frame_ref.image is the BGR frame; res is the FaceFrameResult for that frame.
```

**Output —** `FaceFrameResult`: `boxes (N,4)` xyxy · `scores (N,)` · `landmarks (N,5,2)` ·
`frame_index` · `shape (H,W)`. Coordinates are in the **original** frame. `det.stats.as_dict()`
gives rolling fps / latency / per-stage timings.

### Or from the terminal

```bash
# detect on a video file and show a window (boxes + landmarks + FPS; press q/ESC to quit)
online-face --source video.mp4 --device auto --runtime auto --conf 0.5 --nms 0.4 --display

# webcam (index 0) as a live stream
online-face --source 0 --device auto --runtime auto --stream --display

# headless: write an annotated mp4 instead of showing a window
online-face --source video.mp4 --device auto --runtime auto --save-video out.mp4

# discover weights, or see every flag
online-face --list-weights
online-face --help
```

`online-face` == `python -m online_face.cli.run`. All flags:
`--source` (file path | webcam index | rtsp/http url) · `--device {auto,cpu,cuda,mps}` ·
`--runtime {auto,torch,torchscript,onnx,trt}` · `--conf` · `--nms` · `--stream` · `--display` ·
`--save-video PATH` · `--max-frames N` · `--list-weights`.

---

## Models & weights

`model` is the **family** (`retinaface` — the only one today); `weights` is the actual weight —
a known key (auto-downloaded) or a file path. `weights=None` uses the default.

| weights key | impl | exportable | notes |
|-------------|------|------------|-------|
| `mobilenet0.25` *(default)* | biubug6 | onnx / trt | light, edge-friendly; auto-downloads (~1.7 MB) |
| `resnet50` | biubug6 | onnx / trt | higher accuracy; auto-downloads (~109 MB, sha256-checked) |
| `ternaus_resnet50` | ternaus | torch-only | a convenience weight; works out of the box |

```python
FaceDetector("retinaface", weights="mobilenet0.25")                            # default, auto-downloads
FaceDetector("retinaface", weights="/models/retinaface.onnx", runtime="onnx")  # a ready artifact
```

**`resnet50` is auto-downloaded** (~109 MB, sha256-verified) from the official biubug6 mirror on
first use — nothing to do. If Google Drive ever rate-limits you, download `Resnet50_Final.pth` from
[biubug6/Pytorch_Retinaface](https://drive.google.com/file/d/14KX6VqF69MdSPk3Tr9PlDYbq7ArpdNUW/view) and pass the path:

```python
FaceDetector("retinaface", weights="/path/to/Resnet50_Final.pth")     # or --weights on the CLI/serve
```

…or drop it at `~/.cache/online_inference/weights/retinaface_resnet50.pth` and use `weights="resnet50"`.
(Keep `resnet`/`r50` in the filename — the arch is inferred from the name. The same applies to any
custom weight file.)

---

## Runtimes & the export cache

`runtime="auto"` picks the best backend per device: **Jetson/CUDA → tensorrt** (else onnx-CUDA),
**macOS → torch (MPS)**, **CPU → onnx/torch**. The first time a non-torch runtime is used, the
artifact (torchscript / onnx / trt engine) is **built once and cached** under
`~/.cache/online_inference/` (override with `$ONLINE_INFERENCE_CACHE`); later runs load it.
TensorRT engines are keyed to the exact GPU/JetPack so they never load on the wrong device.

---

## Install options

`[torch]` is all most people need. Add extras for other backends. **Extras are additive** — if
you already installed `[torch]`, running `pip install "online-face-detection[serve]"` later just
adds those packages (it won't reinstall torch). You can also install several at once:
`pip install "online-face-detection[torch,onnx,serve]"`.

| Extra | Adds | Install when you want to… |
|-------|------|---------------------------|
| `[gui]` | opencv-python | desktop OpenCV (enables `cv2.imshow` / `--display`) |
| `[headless]` | opencv-python-headless | OpenCV for servers / Docker / Jetson (no GTK/X11) |
| `[torch]` | torch, torchvision, retinaface-pytorch | **default** runtime (CPU / CUDA / MPS) |
| `[onnx]`  | onnxruntime, onnx, onnxsim | run or export the ONNX backend |
| `[trt]`   | our ONNX export path + fp16 converter (**not** tensorrt) | run the TensorRT backend — install TensorRT yourself first, **see the note below** |
| `[serve]` | fastapi, uvicorn | host the model as an HTTP service (below) |
| `[client]` | requests, websockets | call a remote service (torch-free, below) |

**Which do I actually need?**
- **Always pick an OpenCV build:** `[gui]` (desktop) or `[headless]` (servers/Docker/Jetson). OpenCV is **not** in core, so every real install combines it with a runtime/client extra, e.g. `[torch,gui]` or `[client,headless]`. On Jetson, prefer `[headless]` or rely on JetPack's system OpenCV (install neither).
- `pip install online-face-detection` (no `[...]`) → **core only** (numpy/tqdm); no OpenCV, no runtime — can't run. Use only when OpenCV/torch are provided another way (e.g. JetPack).
- `[torch]` → the **foundation**; required to run the model locally (CPU/CUDA/MPS). Start here (e.g. `[torch,gui]`).
- `[onnx]` / `[trt]` → **add** a backend *on top of* torch (they don't replace it). `[trt]` pulls `[onnx]` + an fp16 converter but **not tensorrt** — install TensorRT for your CUDA yourself (see the note below).
- `[serve]` → runs the model in-process, so it needs torch + an OpenCV build: `pip install "online-face-detection[torch,serve,gui]"`.
- `[client]` → the **only torch-free** path; add an OpenCV build: `pip install "online-face-detection[client,headless]"`.

> [!CAUTION]
> **TensorRT setup.** `[trt]` adds our ONNX export path + fp16 converter but **does not install TensorRT** — its PyPI wheel always grabs the newest CUDA build (e.g. cu13), which won't match your system. Install TensorRT yourself (plus a matching CUDA build of torch). On an NVIDIA machine:
> 1. **Check your CUDA toolkit:** run `nvcc --version` and note the **release** it reports (e.g. `release 12.1`) — that's your installed CUDA toolkit, the version everything below must match. If `nvcc` isn't found, the toolkit isn't installed (install it, or use `[onnx]`/`[torch]` instead).
> 2. **Check torch matches:** `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`. You need a **CUDA build of torch ≥ 2.1** whose CUDA matches step 1 and prints `True`. If not, reinstall it for your CUDA:
>    ```bash
>    pip uninstall -y torch torchvision
>    # pick the matching command from https://pytorch.org/get-started/previous-versions/  (example: CUDA 12.1)
>    pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
>    ```
> 3. **Install TensorRT for your CUDA** following NVIDIA's official guide — pick the build matching step 1, don't rely on a default: <https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-pip.html>
> 4. **Then install this package:**
>    ```bash
>    pip install "online-face-detection[trt]"   # adds our ONNX export path; uses the TensorRT from step 3
>    ```
> On **Jetson**, skip all this — TensorRT ships with JetPack (see [Jetson](#jetson-jetpack)).

---

## (Optional) Serve it as an HTTP service

Besides the in-process use above, the model can run as its own HTTP service (local or cloud)
and be called by URL. Needs the `[serve]` extra (adds only fastapi/uvicorn on top of `[torch]`).

```bash
pip install "online-face-detection[serve]"
online-face-serve --model retinaface --device auto --runtime auto --host 127.0.0.1 --port 8001
```

**Server flags** (all optional; defaults shown): `--model retinaface` ·
`--weights KEY|PATH` (default: family default) · `--device {auto,cpu,cuda,mps}` ·
`--instances 1` · `--runtime {auto,torch,torchscript,onnx,trt}` · `--precision {auto,fp32,fp16,int8}` ·
`--conf 0.5` · `--nms 0.4` · `--input-size N` · `--host 127.0.0.1` · `--port 8001`.

`--instances` runs a pool of N detectors so concurrent requests overlap (each runs in a worker
thread). Pass a number (`--instances 4`) or a per-GPU map (`--instances cuda:0=2,cuda:1=1`) to pin
instances to specific GPUs. On a **single** device the N copies just time-share it (N× memory for
little gain) — multi-instance mainly helps across **multiple GPUs**; the threadpool overlap helps
regardless. `/meta` reports the resolved `instances` and `instance_devices`.

| Route | What it does |
|-------|--------------|
| `GET /meta` | self-describing: named, typed inputs/outputs (input `frame: image`; outputs `boxes/scores/landmarks`); plus `instances`/`instance_devices` |
| `GET /healthz` | readiness + resolved runtime/device |
| `POST /predict` | multipart with a `frame` image part → JSON `{outputs, stats}` (send `Accept: application/x-npz` for a binary array response) |
| `WS /stream` | persistent socket: one binary frame per message → one JSON reply; pipelined, no per-request multipart parsing |

```bash
curl http://127.0.0.1:8001/meta
curl -F 'frame=@frame.png;type=image/png' http://127.0.0.1:8001/predict
```

### Client — `[client]` proxy (torch-free: numpy + requests + websockets, plus an OpenCV build)

```bash
pip install "online-face-detection[client,headless]"   # servers/Jetson; use [client,gui] on desktop
```

```python
from online_face.client import FaceClient

face = FaceClient(
    "http://127.0.0.1:8001",   # service URL (local or cloud)
    encode="jpeg",             # wire format (default): "jpeg" (small, no measurable accuracy loss) | "png" (lossless)
    quality=90,                # JPEG quality (ignored for png)
    max_side=None,             # downscale longest side before sending; boxes still returned in ORIGINAL coords
    binary_response=False,     # True -> .npz array response instead of JSON (skips .tolist(); helps with many boxes)
    timeout=30,
)
```

There are exactly **two** client tools — pick by workload:

| Tool | What it does | Use when |
|------|--------------|----------|
| **`FaceClient`** (unary) — `face(frame)` / `face.predict(frame, max_side=…)` | one request per call, blocking → `FaceResult` (`.boxes/.scores/.landmarks`, original-frame coords); also `healthz()`/`meta()` | one-offs, or a fresh connection per call |
| **`FaceStream`** (async, long-lived) — `await s.push(frame, meta)` + `async for result, meta in s.results()` | holds WebSockets across a **pool of endpoints**, results arrive **as completed (out of order)** tagged with your metadata; auto-scales concurrency to a target fps/latency | continuous / many camera streams, especially over a network |

**Unary knobs:** `encode="jpeg"` (default; ~2–3× smaller than PNG, no measurable accuracy loss — the model letterboxes to a fixed size). `max_side=N` shrinks encode+transfer+decode together (boxes come back rescaled to the original frame). `binary_response=True` skips JSON boxing when there are many faces.

**What you get back** — a `FaceResult` (frozen dataclass), the same from both clients:

| field | type | meaning |
|---|---|---|
| `boxes` | `np.ndarray (N, 4)` float32 | face boxes, **xyxy**, in **original-frame** pixels |
| `scores` | `np.ndarray (N,)` float32 | detection confidence per box |
| `landmarks` | `np.ndarray (N, 5, 2)` float32 | 5 facial points (eyes, nose, mouth corners), original-frame pixels |
| `shape` | `tuple (H, W)` | the original frame size the coords refer to |

`len(result)` == `N` (number of faces). `FaceClient(frame)` returns one `FaceResult`; `FaceStream.results()` yields `(FaceResult, meta)` per frame (your `meta` passed straight through). The in-process `FaceDetector` returns a `FaceFrameResult` — same fields plus `frame_index` and `config`.

### `FaceStream` — continuous, multi-stream, auto-scaling

```python
import asyncio, time
from online_face import FaceStream

async def run(sources):
    # `sources` yields (frame, meta). `meta` is ANY object you choose — it is NOT sent to the
    # server; it's kept client-side and handed back with that frame's result so YOU can tell
    # which output is which. Put whatever you need to route the result, e.g.:
    #     meta = {"stream": cam_id, "i": frame_index, "t": time.time()}
    async with FaceStream(
        ["http://gpu0:8001", "http://gpu1:8001"],  # a single URL, or a POOL of replicas/GPUs
        target_fps=30,            # controller aims for this throughput…
        target_latency_ms=150,    # …while keeping end-to-end latency under this
        max_side=960,             # downscale before sending (boxes still returned in ORIGINAL coords)
        max_inflight=64,          # ceiling on frames in flight across the pool
    ) as stream:

        async def pump():                                  # producer: push frames in
            for frame, meta in sources:                    # frame: (H,W,3) BGR uint8 (or torch CHW)
                await stream.push(frame, meta=meta)        # non-blocking; awaits only under backpressure
            await stream.aclose()                          # end-of-stream (drains in-flight first)
        asyncio.create_task(pump())

        # consumer: results arrive AS COMPLETED — i.e. OUT OF ORDER. Each item is a 2-tuple:
        #     (result: FaceResult, meta)        # `meta` is exactly the object you pushed
        # Identify / reassemble using `meta` — never assume arrival order == push order.
        async for result, meta in stream.results():
            cam, idx = meta["stream"], meta["i"]           # <- how you know which frame this is
            #   result.boxes      -> np.ndarray (N, 4)  xyxy, ORIGINAL-frame pixels
            #   result.scores     -> np.ndarray (N,)    confidence per box
            #   result.landmarks  -> np.ndarray (N,5,2) 5 points per face
            #   result.shape      -> (H, W)             original frame size
            for (x1, y1, x2, y2), score in zip(result.boxes, result.scores):
                handle(cam, idx, x1, y1, x2, y2, score)
            # stream.stats() -> live dict:
            #   {conns, target_inflight, rtt_ms, srv_ms, infer_ms, queue_depth, bound, ...}
```

**How the auto-scaling works.** The session ramps in-flight concurrency up while latency stays under target and throughput keeps rising, and backs off when latency breaches or throughput plateaus. It distinguishes **network-bound** (more sockets / a bigger pool help) from **model-bound** (the server's queue is growing — more inflight just adds latency) using server-reported `infer_ms`/`queue_depth` vs measured RTT. Give it **one URL** and it tunes concurrency against that single model (it can't exceed one model's throughput); give it a **pool** and it scales out across replicas/GPUs and holds when all are saturated.

> Same device? Skip HTTP entirely and call the in-process `FaceDetector`.

### Faster inference: bake decode + NMS into the graph (`postprocess="graph"`)

RetinaFace post-processing (prior decode + NMS) costs about as much as inference itself. With an exported runtime you can fold it into the graph so it runs in the engine, not Python — detections are identical to the raw path:

```python
det = FaceDetector("retinaface", runtime="onnx", postprocess="graph", conf=0.5, nms=0.4)
# or:  online-face-serve --runtime onnx --postprocess graph
```
Notes: `conf`/`nms`/`max_faces` are **baked at export time** (changing them re-exports); the artifact is **fixed to one input size**. **`onnx` is verified.** **`trt` is experimental/unverified** — it parses the ONNX NonMaxSuppression graph, which TRT support varies for; validate on your GPU (for production, prefer raw export or an `EfficientNMS_TRT` plugin graph). **Not** available on the eager `torch` runtime (no graph) or `torchscript` (backbone isn't scriptable). Raw export remains the default.

---

## Misc

### Composing face → emotion (no combined package)

The two packages stay independent — there is intentionally no combined package; the glue is two lines of your code.

```python
# In-process (same device) — fastest; crops never leave the device:
from online_face import FaceDetector
from online_emotion import EmotionRecognizer
det, emo = FaceDetector("retinaface"), EmotionRecognizer("hsemotion")
r = det(frame); emotions = emo.predict_on_boxes(frame, r.boxes)

# Over the wire (two services) — send only face crops to emotion, not the whole frame twice:
from online_face.client import FaceClient
from online_emotion.client import EmotionClient
face, emo = FaceClient(FACE_URL), EmotionClient(EMO_URL)
r = face(frame)
emotions = emo.predict_on_crops(EmotionClient.crop_boxes(frame, r.boxes))
```

### Install with uv

Same as pip, with `uv` (include an OpenCV build — `gui` or `headless`):
```bash
uv add "online-face-detection[torch,gui]"            # into a uv project
uv pip install "online-face-detection[torch,headless]"    # into the active venv
```

### Jetson (JetPack)

On Jetson the whole GPU stack (CUDA / cuDNN / TensorRT) is part of **JetPack**, and torch/onnxruntime
must be NVIDIA's Jetson wheels — the PyPI `[torch]`/`[onnx]` wheels are x86_64 and won't use the GPU.

**1. Pick a JetPack version.**

| Board | JetPack | Stack |
|-------|---------|-------|
| Orin (AGX/NX/Nano) | **6.x** | CUDA 12.6 · TensorRT 10.3 · PyTorch 2.6 wheel |
| Xavier / older | **5.1.x** | torch ~2.1 |

Both are above this package's `torch>=2.1` floor.

**2. Install these into the JetPack env first** — from NVIDIA's
[PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) guide, or the
[jetson-ai-lab](https://pypi.jetson-ai-lab.io) wheel index matched to your JetPack (e.g.
`--index-url https://pypi.jetson-ai-lab.io/jp6/cu126` for JetPack 6.x):

- `torch`, `torchvision` — the **Jetson GPU wheels** (not from PyPI)
- `onnxruntime-gpu` — only if you'll use the ONNX backend
- `opencv-python`, `numpy` — usually already present in JetPack; install if missing
- TensorRT — **already installed by JetPack** (nothing to do)

**3. Then install this package with NO runtime extra**, so it uses the system ones:

```bash
pip install online-face-detection      # no [torch] / [onnx]
```

It adapts to whatever JetPack provides and keys each cached TensorRT engine to the exact board.

> **Conflicting model requirements?** One Jetson has a single system torch/TRT. If two models need
> incompatible torch/CUDA, run each as its own [HTTP service](#optional-serve-it-as-an-http-service)
> (e.g. an `nvcr.io/nvidia/l4t-pytorch` container) and compose them by URL with the `[client]` proxy.

### Pre-build & cache an artifact

Optional — otherwise built on first use. Choose the runtime you'll deploy with for the target device:
```bash
online-face-export --model retinaface --weights mobilenet0.25 --runtime trt --device auto
```
Flags: `--model` · `--weights KEY|PATH` · `--runtime {torchscript,onnx,trt}` ·
`--device {auto,cpu,cuda,mps}` · `--precision {auto,fp32,fp16,int8}` · `--input-size N`.

## License

MIT © Surya Chand Rayala
