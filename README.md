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
| `[torch]` | torch, torchvision, retinaface-pytorch | **default** runtime (CPU / CUDA / MPS) |
| `[onnx]`  | onnxruntime, onnx, onnxsim | run or export the ONNX backend |
| `[trt]`   | our ONNX export path + fp16 converter (**not** tensorrt) | run the TensorRT backend — install TensorRT yourself first, **see the note below** |
| `[serve]` | fastapi, uvicorn | host the model as an HTTP service (below) |
| `[client]` | requests | call a remote service (torch-free, below) |

**Which do I actually need?**
- `pip install online-face-detection` (no `[...]`) → **core only** (numpy/opencv); **no runtime, can't run inference**. Use this only when torch is provided another way (e.g. Jetson/JetPack wheels).
- `[torch]` → the **foundation**; required to run the model locally (CPU/CUDA/MPS). Start here.
- `[onnx]` / `[trt]` → **add** a backend *on top of* torch (they don't replace it). `[trt]` pulls `[onnx]` + an fp16 converter but **not tensorrt** — install TensorRT for your CUDA yourself (see the note below).
- `[serve]` → runs the model in-process, so it needs torch too: `pip install "online-face-detection[torch,serve]"`.
- `[client]` → the **only torch-free** one — it just calls a remote service, so `pip install "online-face-detection[client]"` **alone is enough**.

> [!CAUTION]
> **TensorRT setup.** `[trt]` adds our ONNX export path + fp16 converter but **does not install TensorRT** — its PyPI wheel always grabs the newest CUDA build (e.g. cu13), which won't match your system. Install TensorRT yourself (plus a matching CUDA build of torch). On an NVIDIA machine:
> 1. **Check your CUDA:** run `nvidia-smi` and note the **CUDA Version** it reports (your GPU/driver's CUDA). No NVIDIA GPU → use `[onnx]`/`[torch]` instead.
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
`--runtime {auto,torch,torchscript,onnx,trt}` · `--precision {auto,fp32,fp16,int8}` ·
`--conf 0.5` · `--nms 0.4` · `--input-size N` · `--host 127.0.0.1` · `--port 8001`.

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

face = FaceClient(
    "http://127.0.0.1:8001",   # the service URL (local or cloud)
    encode="png",              # how frames go over the wire: "png" (lossless) | "jpeg" (smaller)
    timeout=30,                # request timeout, seconds
)
res = face(frame)              # same shape as det(frame): res.boxes / res.scores / res.landmarks
face.meta()                    # the service's /meta;  face.healthz() -> readiness
```

Compose two services into a pipeline (e.g. face → emotion) by URL — see
**[../testing-pipeline](../testing-pipeline)** for a ready-to-run example.

---

## Misc

### Install with uv

Same as pip, with `uv`:
```bash
uv add "online-face-detection[torch]"            # into a uv project
uv pip install "online-face-detection[torch]"    # into the active venv
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
