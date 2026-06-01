"""Minimal demo: frame-by-frame and stream usage with terminal FPS stats.

    python examples/demo.py --source /path/to/video.mp4 --display
    python examples/demo.py --source 0 --stream --display        # webcam
"""
from __future__ import annotations

import argparse

from online_face import FaceDetector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video path | webcam index | rtsp/http url")
    ap.add_argument("--stream", action="store_true", help="treat source as a live stream")
    ap.add_argument("--display", action="store_true")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--runtime", default="auto")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    # One detector object, reused frame-by-frame (this is what you embed in a pipeline).
    det = FaceDetector("retinaface", weights=args.weights, runtime=args.runtime, device="auto")
    print("config:", det.config.summary())

    for frame_ref, result in det.run_source(args.source, is_stream=(True if args.stream else None),
                                             display=args.display, max_frames=args.max_frames):
        # result.boxes (N,4), result.scores (N,), result.landmarks (N,5,2) in original-frame coords.
        pass

    print("final stats:", det.stats.as_dict())
    det.close()


if __name__ == "__main__":
    main()
