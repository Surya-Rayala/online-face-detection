# online-face-detection

Streaming, **frame-by-frame** face detection (RetinaFace) for real-time pipelines:
one small object — a frame in, structured results out. Runs under **torch / torchscript /
onnx / tensorrt** with export-once caching, on CPU, CUDA, Apple Silicon (MPS), and Jetson.

```python
from online_face import FaceDetector
det = FaceDetector("retinaface", device="auto")
res = det(frame)        # res.boxes, res.scores, res.landmarks
```

---

## Install

```bash
pip install "online-face-detection[torch]"
```

That's all you need for most setups — `[torch]` is the default runtime and works on CPU,
CUDA, and Mac (MPS). Other backends (`onnx`, `tensorrt`, serving) are **optional extras** —
see [Install options](#install-options) below. (Prefer `uv`? See [Misc](#misc).)

---

## Use it

### In Python

```python
from online_face import FaceDetector

det = FaceDetector("retinaface", device="auto")   # auto -> CUDA / MPS / CPU; weights auto-download
res = det(frame)                                   # frame: BGR ndarray (HWC) or torch.Tensor

res.boxes        # (N, 4) xyxy, in original-frame coordinates
res.scores       # (N,)
res.landmarks    # (N, 5, 2)
```

Drive a video file or a live stream (FPS/latency print to the terminal):

```python
for frame_ref, res in det.run_source("video.mp4"):                    # a file
    ...
for frame_ref, res in det.run_source("rtsp://cam/stream", is_stream=True):   # live
    ...
for frame_ref, res in det.run_source("video.mp4", is_stream=True):    # file replayed as a stream
    ...
```

### From the terminal

```bash
online-face --source video.mp4 --display               # window with boxes/landmarks + FPS (q/ESC quits)
online-face --source 0 --stream --display              # webcam
online-face --source video.mp4 --save-video out.mp4    # headless: write an annotated mp4
online-face --list-weights                             # list available weights
```

`online-face` is the same as `python -m online_face.cli.run`. Useful flags:
`--runtime {auto,torch,torchscript,onnx,trt}` · `--device {auto,cpu,cuda,mps}` ·
`--conf` · `--nms` · `--max-frames`.

### Output

`FaceFrameResult`: `boxes (N,4)` xyxy · `scores (N,)` · `landmarks (N,5,2)` · `frame_index` ·
`shape (H,W)`. Coordinates are in the **original** frame. `det.stats.as_dict()` gives rolling
fps / latency / per-stage timings.

---

## Models & weights

`model` is the **family** (`retinaface`); `weights` is the actual weight — a known key
(auto-downloaded) or a file path. `weights=None` uses the default.

| weights key | impl | exportable | notes |
|-------------|------|------------|-------|
| `mobilenet0.25` *(default)* | biubug6 | onnx / trt | light, edge-friendly; auto-downloads (~1.7 MB) |
| `resnet50` | biubug6 | onnx / trt | higher accuracy; weights placed manually |
| `ternaus_resnet50` | ternaus | torch-only | convenience; needs the `[ternaus]` extra |

```python
FaceDetector("retinaface", weights="resnet50")
FaceDetector("retinaface", weights="/models/retinaface.onnx", runtime="onnx")  # a ready artifact
```

---

## Serve it as an HTTP service

Host the model as its own service (local or cloud) and call it by URL. Needs the `[serve]` extra.

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

## Use it remotely (client)

`[client]` is a **torch-free** proxy (`requests` + numpy + opencv) mirroring the local call shape:

```bash
pip install "online-face-detection[client]"
```

```python
from online_face.client import FaceClient

face = FaceClient("http://127.0.0.1:8001")   # or a cloud URL
res = face(frame)                            # same shape as det(frame)
```

Compose two services into a pipeline (e.g. face → emotion) just by URL — see
**[../testing-pipeline](../testing-pipeline)** for a ready-to-run example.

---

## Runtimes & the export cache

`runtime="auto"` picks the best backend per device: **Jetson/CUDA → tensorrt** (else onnx-CUDA),
**macOS → torch (MPS)**, **CPU → onnx/torch**. The first time a non-torch runtime is used, the
artifact (torchscript / onnx / trt engine) is **built once and cached** under
`~/.cache/online_inference/` (override with `$ONLINE_INFERENCE_CACHE`); later runs load it.
TensorRT engines are keyed to the exact GPU/JetPack so they never load on the wrong device.

---

## Install options

`[torch]` is all most people need. Add extras for other backends (combine freely, e.g.
`pip install "online-face-detection[torch,onnx,serve]"`):

| Extra | Adds | Install when you want to… |
|-------|------|---------------------------|
| `[torch]` | torch, torchvision | **default** runtime (CPU / CUDA / MPS) |
| `[onnx]`  | onnxruntime, onnx, onnxsim | run or export the ONNX backend |
| `[trt]`   | tensorrt | build/run TensorRT engines (NVIDIA) |
| `[ternaus]` | retinaface-pytorch | use the `ternaus_resnet50` weight |
| `[serve]` | fastapi, uvicorn | host the model as an HTTP service |
| `[client]` | requests | call a remote service (torch-free) |

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
