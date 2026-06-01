"""Public face-detection API: ``FaceDetector`` + ``FaceFrameResult``.

Frame in -> structured result out, one frame per call. The same object drives a
file or a live/simulated stream via :meth:`run_source`, which can overlay
detections (``display``) and prints FPS/latency to the terminal as it runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple, Union

import numpy as np

from .families import available_models as _available_models
from .families import get_family
from .runtime.config import InferenceConfig
from .runtime.sources import FrameRef, open_source
from .runtime.streaming import StreamingModel
from .runtime.tensor import load_frame
from .runtime.timing import Stopwatch
from .runtime.viz import draw_box_label, draw_hud, draw_points, hash_color

Frame = Union[np.ndarray, "object"]  # numpy BGR HWC, or torch.Tensor


@dataclass(frozen=True)
class FaceFrameResult:
    boxes: np.ndarray         # (N, 4) xyxy in ORIGINAL frame coords
    scores: np.ndarray        # (N,)
    landmarks: np.ndarray     # (N, 5, 2)
    frame_index: int
    shape: Tuple[int, int]    # (H, W) the coords refer to
    config: InferenceConfig

    def __len__(self) -> int:
        return int(self.boxes.shape[0])


class FaceDetector(StreamingModel):
    package = "online_face"

    def __init__(self, model: str = "retinaface", *, weights=None, runtime: str = "auto",
                 device="auto", precision: str = "auto", conf: float = 0.5, nms: float = 0.4,
                 input_size=None, cache_dir=None, warmup: bool = True, display: bool = False) -> None:
        self.conf = float(conf)
        self.nms = float(nms)
        super().__init__(family=get_family(model), weights=weights, runtime=runtime, device=device,
                         precision=precision, input_size=input_size, cache_dir=cache_dir,
                         warmup=warmup, display=display)

    # -- single frame ------------------------------------------------------
    def predict(self, frame: Frame, *, frame_index: Optional[int] = None) -> FaceFrameResult:
        params = {"conf": self.conf, "nms": self.nms}
        if self.is_native:
            img = frame if isinstance(frame, np.ndarray) else self._to_bgr_numpy(frame)
            with Stopwatch() as t_inf:
                payload = self._family.native_predict(self._native_model, img, params)
            h, w = img.shape[:2]
            self.stats.update(0.0, t_inf.elapsed, 0.0)
        else:
            frame_t = load_frame(frame, self._device)
            h, w = int(frame_t.shape[1]), int(frame_t.shape[2])
            with Stopwatch() as t_pre:
                inp, ctx = self._family.preprocess(frame_t, self._input_size)
            with Stopwatch() as t_inf:
                raw = self._infer(inp)
            with Stopwatch() as t_post:
                payload = self._family.postprocess(raw, ctx, params)
            self.stats.update(t_pre.elapsed, t_inf.elapsed, t_post.elapsed)
        idx = frame_index if frame_index is not None else (self.stats.frames - 1)
        return FaceFrameResult(payload["boxes"], payload["scores"], payload["landmarks"],
                               frame_index=idx, shape=(h, w), config=self._config)

    __call__ = predict

    @staticmethod
    def _to_bgr_numpy(frame) -> np.ndarray:
        t = load_frame(frame, "cpu")
        return t.permute(1, 2, 0).round().clamp(0, 255).byte().numpy()

    # -- stream / file -----------------------------------------------------
    def run_source(self, source, *, is_stream: Optional[bool] = None, display: Optional[bool] = None,
                   max_frames: Optional[int] = None, save_video: Optional[str] = None,
                   print_stats: bool = True) -> Iterator[Tuple[FrameRef, FaceFrameResult]]:
        show = self.display if display is None else display
        src = open_source(source, is_stream=is_stream, max_frames=max_frames)
        every = max(1, int(src.fps or 30))
        writer = None
        win = "online_face"
        try:
            for fref in src:
                res = self.predict(fref.image, frame_index=fref.index)
                if show or save_video:
                    canvas = fref.image.copy()
                    self._draw(canvas, res)
                    draw_hud(canvas, self.stats.format_line())
                    if save_video:
                        if writer is None:
                            import cv2

                            h, w = canvas.shape[:2]
                            writer = cv2.VideoWriter(str(save_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                                     src.fps or 30.0, (w, h))
                        writer.write(canvas)
                    if show:
                        import cv2

                        cv2.imshow(win, canvas)
                        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                            break
                self.stats.dropped = src.dropped
                if print_stats and fref.index % every == 0:
                    print(f"\r[online_face] {self.stats.format_line()} faces={len(res):<3}", end="", flush=True)
                yield fref, res
        finally:
            if print_stats:
                print(f"\r[online_face] {self.stats.format_line()} (done){' ' * 12}")
            if writer is not None:
                writer.release()
            if show:
                try:
                    import cv2

                    cv2.destroyWindow(win)
                except Exception:
                    pass
            src.release()

    def _draw(self, img: np.ndarray, res: FaceFrameResult) -> None:
        for i, (box, score) in enumerate(zip(res.boxes, res.scores)):
            draw_box_label(img, box, hash_color(i), f"{score:.2f}")
        for lm in res.landmarks:
            draw_points(img, lm, (0, 255, 255), 2)

    # -- export / discovery ------------------------------------------------
    @classmethod
    def export(cls, model: str = "retinaface", *, weights=None, runtime: str, device="auto",
               precision: str = "auto", input_size=None, cache_dir=None) -> InferenceConfig:
        """Build/cache the artifact for ``runtime`` and return the resolved config."""
        det = cls(model, weights=weights, runtime=runtime, device=device, precision=precision,
                  input_size=input_size, cache_dir=cache_dir, warmup=False)
        cfg = det.config
        det.close()
        return cfg

    @staticmethod
    def available_models() -> list:
        return _available_models()

    @staticmethod
    def available_weights(model: str = "retinaface") -> list:
        return get_family(model).available_weights()
