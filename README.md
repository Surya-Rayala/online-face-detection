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

```bash
pip install "online-face-detection[torch]"
```

That's all you need for most setups — `[torch]` is the default runtime and works on CPU,
CUDA, and Mac (MPS). Other backends (`onnx`, `tensorrt`, serving) are **optional extras** you
can add anytime — see [Install options](#install-options). (Prefer `uv`? See [Misc](#misc).)

---

## Use it (Python)

```python
from online_face import FaceDetector

det = FaceDetector("retinaface", device="auto")   # auto -> CUDA / MPS / CPU; weights auto-download
res = det(frame)

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
online-face --source video.mp4 --display               # window with boxes/landmarks + FPS (q/ESC quits)
online-face --source 0 --stream --display              # webcam
online-face --source video.mp4 --save-video out.mp4    # headless: write an annotated mp4
online-face --list-weights
```

`online-face` == `python -m online_face.cli.run`. Useful flags:
`--runtime {auto,torch,torchscript,onnx,trt}` · `--device {auto,cpu,cuda,mps}` ·
`--conf` · `--nms` · `--max-frames`.

---

## Models & weights

`model` is the **family** (`retinaface` — the only one today); `weights` is the actual weight —
a known key (auto-downloaded) or a file path. `weights=None` uses the default.

| weights key | impl | exportable | notes |
|-------------|------|------------|-------|
| `mobilenet0.25` *(default)* | biubug6 | onnx / trt | light, edge-friendly; auto-downloads (~1.7 MB) |
| `resnet50` | biubug6 | onnx / trt | higher accuracy; weights placed manually |
| `ternaus_resnet50` | ternaus | torch-only | works out of the box (bundled with `[torch]`) |

```python
FaceDetector("retinaface", weights="resnet50")
FaceDetector("retinaface", weights="/models/retinaface.onnx", runtime="onnx")  # a ready artifact
```

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
| `[torch]` | torch, torchvision, retinaface-pytorch | **default** runtime (CPU / CUDA / MPS) |
| `[onnx]`  | onnxruntime, onnx, onnxsim | run or export the ONNX backend |
| `[trt]`   | tensorrt | build/run TensorRT engines (NVIDIA) |
| `[serve]` | fastapi, uvicorn | host the model as an HTTP service (below) |
| `[client]` | requests | call a remote service (torch-free, below) |

---

## (Optional) Serve it as an HTTP service

Besides the in-process use above, the model can run as its own HTTP service (local or cloud)
and be called by URL. Needs the `[serve]` extra (adds only fastapi/uvicorn on top of `[torch]`).

```bash
pip install "online-face-detection[serve]"
online-face-serve --runtime torch --device mps --port 8001
```

| Route | What it does |
|-------|--------------|
| `GET /meta` | self-describing: named, typed inputs/outputs (input `frame: image`; outputs `boxes/scores/landmarks`) |
| `GET /healthz` | readiness + resolved runtime/device |
| `POST /predict` | multipart with a `frame` image part → JSON `{outputs, stats}` |

```bash
curl http://127.0.0.1:8001/meta
curl -F 'frame=@frame.png;type=image/png' http://127.0.0.1:8001/predict
```

**Call it from another process** with the torch-free `[client]` proxy (mirrors `det(frame)`):

```bash
pip install "online-face-detection[client]"
```
```python
from online_face.client import FaceClient
face = FaceClient("http://127.0.0.1:8001")   # or a cloud URL
res = face(frame)                            # same shape as det(frame)
```

Compose two services into a pipeline (e.g. face → emotion) by URL — see
**[../testing-pipeline](../testing-pipeline)** for a ready-to-run example.

---

## Misc

**Install with uv** instead of pip:
```bash
uv add "online-face-detection[torch]"            # into a uv project
uv pip install "online-face-detection[torch]"    # into the active venv
```

**Jetson:** install torch / onnxruntime-gpu / tensorrt from the JetPack/NVIDIA wheels (don't
`pip install` them), then `pip install online-face-detection` (without `[torch]`).

**Pre-build an artifact** (optional — otherwise built on first use):
```bash
online-face-export --model retinaface --runtime onnx
```

## License

MIT © Surya Chand Rayala
