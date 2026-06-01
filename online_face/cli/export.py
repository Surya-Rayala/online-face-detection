"""``python -m online_face.cli.export`` — pre-export & cache an artifact.

Builds the torchscript/onnx/trt artifact once and caches it; later runs with
the same config load it instead of re-exporting.
"""
from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    from .. import FaceDetector

    p = argparse.ArgumentParser("online_face.cli.export", description="Export and cache a model artifact.")
    p.add_argument("--model", default="retinaface")
    p.add_argument("--weights", default=None)
    p.add_argument("--runtime", required=True, choices=["torchscript", "onnx", "trt"])
    p.add_argument("--device", default="auto")
    p.add_argument("--precision", default="auto", choices=["auto", "fp32", "fp16", "int8"])
    p.add_argument("--input-size", type=int, default=None)
    args = p.parse_args(argv)

    cfg = FaceDetector.export(args.model, weights=args.weights, runtime=args.runtime,
                              device=args.device, precision=args.precision, input_size=args.input_size)
    print(json.dumps(cfg.to_dict(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
