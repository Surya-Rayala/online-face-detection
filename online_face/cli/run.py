"""``python -m online_face.cli.run`` — run detection on a video/stream.

Prints rolling FPS/latency to the terminal and a JSON summary at the end.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("online_face.cli.run",
                                description="Streaming face detection on a video file or live stream.")
    p.add_argument("--source", help="video path | webcam index (e.g. 0) | rtsp/http url")
    p.add_argument("--model", default="retinaface")
    p.add_argument("--weights", default=None, help="weight key (auto-download) or a local file path")
    p.add_argument("--runtime", default="auto", choices=["auto", "torch", "torchscript", "onnx", "trt"])
    p.add_argument("--device", default="auto")
    p.add_argument("--precision", default="auto", choices=["auto", "fp32", "fp16", "int8"])
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--nms", type=float, default=0.4)
    p.add_argument("--input-size", type=int, default=None)
    p.add_argument("--stream", action="store_true",
                   help="treat --source as a live stream (real-time pacing, drop-to-latest)")
    p.add_argument("--display", action="store_true", help="show an overlay window (press q/ESC to quit)")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--save-video", default=None, help="write an annotated mp4")
    p.add_argument("--list-models", action="store_true")
    p.add_argument("--list-weights", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    from .. import FaceDetector

    args = build_parser().parse_args(argv)
    if args.list_models:
        print("\n".join(FaceDetector.available_models()))
        return 0
    if args.list_weights:
        print("\n".join(FaceDetector.available_weights(args.model)))
        return 0
    if not args.source:
        print("error: --source is required (or use --list-models / --list-weights)", file=sys.stderr)
        return 2

    det = FaceDetector(args.model, weights=args.weights, runtime=args.runtime, device=args.device,
                       precision=args.precision, conf=args.conf, nms=args.nms,
                       input_size=args.input_size, display=args.display)
    frames = 0
    for _fref, _res in det.run_source(args.source, is_stream=(True if args.stream else None),
                                      display=args.display, max_frames=args.max_frames,
                                      save_video=args.save_video):
        frames += 1
    print(json.dumps({"frames": frames, "stats": det.stats.as_dict(), "config": det.config.to_dict()},
                     indent=2, default=str))
    det.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
