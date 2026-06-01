# online-face-detection

Streaming, **frame-by-frame** face detection you drop into a real-time pipeline. One small object: a frame (NumPy **or** torch tensor) goes in, a structured result comes out. Unified runtime selector (**torch / torchscript / onnx / trt** + `auto`) and **export-once-then-reuse** caching for edge devices (Jetson).

First model family: **RetinaFace** (biubug6 MobileNet0.25 / ResNet50, plus a ternaus convenience path). More families plug in via a registry.

## Install & use in 30 seconds

**pip:**
```bash
pip install "online-face-detection[torch]"      # desktop / macOS (MPS) — the default
```
**uv:**
```bash
uv add "online-face-detection[torch]"           # add to a uv project
uv pip install "online-face-detection[torch]"   # or into the active venv
```
**then in Python:**
```python
from online_face import FaceDetector

det = FaceDetector("retinaface", device="auto")   # auto -> CUDA/MPS/CPU
res = det(frame)                                   # BGR ndarray or torch.Tensor
res.boxes, res.scores, res.landmarks               # (N,4), (N,), (N,5,2)
```

| Extra | Adds | When |
|-------|------|------|
| `[torch]` | torch, torchvision | **default** runtime (CPU / CUDA / MPS) |
| `[onnx]`  | onnxruntime, onnx, onnxsim | ONNX runtime + export |
| `[trt]`   | tensorrt | TensorRT engines (discrete NVIDIA) |
| `[ternaus]` | retinaface-pytorch | the `ternaus_resnet50` convenience weight |

> **Jetson:** install torch / onnxruntime-gpu / tensorrt from the JetPack/NVIDIA wheels (don't `pip install` them), then `pip install online-face-detection`.

## Quickstart

```python
from online_face import FaceDetector

det = FaceDetector("retinaface", device="auto")     # auto -> CUDA/MPS/CPU; default weights auto-download
res = det(frame)                                     # frame: BGR ndarray (HWC) or torch.Tensor
res.boxes        # (N, 4) xyxy in original-frame coords
res.scores       # (N,)
res.landmarks    # (N, 5, 2)
```

Drive a **file or a live/simulated stream** (FPS/latency print to the terminal as it runs):

```python
for frame_ref, res in det.run_source("video.mp4", display=True):     # static file
    ...
for frame_ref, res in det.run_source("rtsp://cam/stream", is_stream=True):   # live
    ...
for frame_ref, res in det.run_source("video.mp4", is_stream=True):   # file replayed as a stream
    ...
```

`display=True` overlays boxes/landmarks + a FPS HUD (q/ESC to quit) — for verification; leave it **off** in production.

## CLI

```bash
python -m online_face.cli.run --source video.mp4 --display            # or: online-face --source ...
python -m online_face.cli.run --source 0 --stream --display           # webcam
python -m online_face.cli.run --source video.mp4 --runtime onnx --save-video out.mp4
python -m online_face.cli.run --list-models
python -m online_face.cli.run --list-weights
```

## Serve it over HTTP (host the model as a service)

The model can run as its own HTTP service — a local process or a remote/cloud
instance — so a pipeline can call it by URL. It speaks a small **uniform contract**
(the same one every model in this project uses), so one generic client can talk to
any of them.

```bash
pip install "online-face-detection[serve]"        # adds fastapi + uvicorn (no extra torch)
online-face-serve --runtime torch --device mps --port 8001
#  or: python -m online_face serve --runtime onnx --port 8001
```

Endpoints:

| Route | What it does |
|-------|--------------|
| `GET /meta` | self-describing: `name`, `modality`, and the **named, typed** `inputs`/`outputs` (here: input `frame` of type `image`; outputs `boxes`/`scores`/`landmarks` as `ndarray`) |
| `GET /healthz` | readiness + resolved `runtime`/`device`/`mps` (ready only **after** the model — and any first-run export — is built) |
| `POST /predict` | `multipart/form-data` with a `frame` image part → JSON `{outputs, stats}` |

```bash
curl http://127.0.0.1:8001/meta
# /predict takes a multipart 'frame' part (content-type tags the payload kind):
curl -F 'frame=@frame.png;type=image/png' http://127.0.0.1:8001/predict
```

The service builds the model **once** at startup (so the export-once cache is warm
before traffic) and runs single-process — keep one worker (the MPS/CUDA model lives
in-process).

## Call it remotely (the client proxy) & build pipelines

`[client]` is a **torch-free** proxy (just `requests` + numpy + opencv) that mirrors
the local `FaceDetector` call shape — so remote code reads like in-process code:

```bash
pip install "online-face-detection[client]"       # torch-free: deploy anywhere
```

```python
from online_face.client import FaceClient

face = FaceClient("http://127.0.0.1:8001")         # or a cloud URL
res = face(frame)                                  # same shape as det(frame)
res.boxes, res.scores, res.landmarks               # numpy arrays
```

Because each model is just a URL, **composing models into a pipeline is trivial** —
host a face service and an emotion service (each in its own env), then:

```python
from online_face.client import FaceClient
from online_emotion.client import EmotionClient

face = FaceClient("http://127.0.0.1:8001")
emo  = EmotionClient("http://127.0.0.1:8002")
r  = face(frame)                                   # detect
er = emo.predict_on_boxes(frame, r.boxes)          # recognise emotion on the boxes
```

For a ready-to-run face→emotion test on your own video (turn the services on, then run
one script), see **[../testing-pipeline](../testing-pipeline)**.

## Models & weights

`model` is the **family**; `weights` is the actual weight — a known key (auto-downloaded) **or** a file path. `weights=None` uses the family default.

| weights key | impl | exportable | notes |
|-------------|------|------------|-------|
| `mobilenet0.25` *(default)* | biubug6 | ✅ onnx/trt | light, edge-friendly; auto-downloads (~1.7 MB) |
| `resnet50` | biubug6 | ✅ onnx/trt | higher accuracy; weights placed manually (clear instructions on first use) |
| `ternaus_resnet50` | ternaus | ❌ torch-only | convenience; needs `[ternaus]` |

```python
FaceDetector("retinaface", weights="resnet50")
FaceDetector("retinaface", weights="/models/retinaface.onnx", runtime="onnx")   # a ready artifact loads directly
```

## Runtimes, devices & the export cache

`runtime="auto"` picks per device: **Jetson/CUDA → trt** (else onnx-CUDA), **macOS → torch (MPS)**, **CPU → onnx/torch**. `precision="auto"` → fp16 on CUDA/TRT, fp32 on MPS/CPU.

The first time a non-torch runtime is requested, the artifact (torchscript/onnx/trt engine) is **built once and cached** under `~/.cache/online_inference/artifacts/<key>/` (override with `$ONLINE_INFERENCE_CACHE`); later runs **load the cache**. TensorRT engines are keyed to the exact GPU/JetPack so they never load on the wrong device.

```bash
python -m online_face.cli.export --model retinaface --weights mobilenet0.25 --runtime trt --precision fp16 --input-size 640
```

## Output

`FaceFrameResult(boxes (N,4) xyxy, scores (N,), landmarks (N,5,2), frame_index, shape=(H,W), config)` — coordinates are in the **original** frame. `det.stats.as_dict()` gives rolling fps / latency / per-stage timings.

## License

MIT © Surya Chand Rayala
