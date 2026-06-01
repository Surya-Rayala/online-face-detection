"""RetinaFace family: biubug6 (default, exportable) + ternaus (native, torch-only).

The weight key selects the implementation: ``mobilenet0.25`` / ``resnet50`` use
the clean biubug6 graph (shared decode/NMS, exports to onnx/trt), while
``ternaus_resnet50`` delegates to the ``retinaface-pytorch`` package's own
pipeline (a convenience path, torch only).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .. import registry as _registry
from ..models.retinaface import build_retinaface, decode, decode_landm, prior_box
from ..runtime.errors import RuntimeUnavailableError
from ..runtime.logging import get_logger
from ..runtime.tensor import letterbox, rescale_boxes, rescale_points
from .base import ExportSpec, ModelFamily, ResolvedWeights

_log = get_logger("retinaface")
_MEAN_BGR = (104.0, 117.0, 123.0)
_VARIANCE = (0.1, 0.2)


class RetinaFaceFamily(ModelFamily):
    name = "retinaface"
    package = "online_face"
    default_input_size = (640, 640)

    def __init__(self) -> None:
        self._prior_cache: Dict[Any, Any] = {}

    # -- discovery ---------------------------------------------------------
    def available_weights(self) -> List[str]:
        return _registry.available_weights()

    def resolve_weights(self, weights: Optional[str], cache) -> ResolvedWeights:
        return _registry.resolve_weights(weights, cache)

    # -- biubug6 graph -----------------------------------------------------
    def build_module(self, resolved: ResolvedWeights, device: str, precision: str):
        import torch

        if resolved.meta.get("native"):
            raise RuntimeUnavailableError("ternaus weights run natively; build_module is not used")
        model, _ = build_retinaface(resolved.arch)
        sd = torch.load(str(resolved.path), map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            _log.debug("missing keys when loading %s: %d", resolved.key, len(missing))
        return model.eval()

    def preprocess(self, frame_chw_bgr, input_size: Tuple[int, int]):
        import torch

        out, meta = letterbox(frame_chw_bgr, input_size, pad_value=_MEAN_BGR)
        mean = torch.tensor(_MEAN_BGR, device=out.device, dtype=out.dtype).view(3, 1, 1)
        out = out - mean
        return out.unsqueeze(0), {"meta": meta, "input_size": input_size}

    def _priors(self, input_size: Tuple[int, int], device: str):
        key = (input_size, str(device))
        if key not in self._prior_cache:
            self._prior_cache[key] = prior_box(input_size, device)
        return self._prior_cache[key]

    def postprocess(self, raw, ctx, params: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        from torchvision.ops import nms

        loc, conf, landms = raw[0], raw[1], raw[2]
        meta = ctx["meta"]
        h_in, w_in = ctx["input_size"]
        priors = self._priors(ctx["input_size"], loc.device)

        boxes = decode(loc[0].float(), priors, _VARIANCE)
        boxes = boxes * torch.tensor([w_in, h_in, w_in, h_in], device=boxes.device, dtype=boxes.dtype)
        lm = decode_landm(landms[0].float(), priors, _VARIANCE)
        lm = lm * torch.tensor([w_in, h_in] * 5, device=lm.device, dtype=lm.dtype)
        scores = conf[0].float()[:, 1]

        keep = scores > float(params.get("conf", 0.5))
        boxes, scores, lm = boxes[keep], scores[keep], lm[keep]
        if boxes.shape[0] > 0:
            order = nms(boxes, scores, float(params.get("nms", 0.4)))[: int(params.get("max_faces", 2000))]
            boxes, scores, lm = boxes[order], scores[order], lm[order]
            boxes = rescale_boxes(boxes, meta)
            lm = rescale_points(lm.view(-1, 5, 2), meta)
        else:
            lm = lm.view(-1, 5, 2)
        return {
            "boxes": boxes.detach().cpu().numpy(),
            "scores": scores.detach().cpu().numpy(),
            "landmarks": lm.detach().cpu().numpy(),
        }

    def export_spec(self, input_size: Tuple[int, int]) -> ExportSpec:
        return ExportSpec(input_names=["input"], output_names=["loc", "conf", "landms"],
                          dynamic_axes=None, opset=13)

    def resolve_input_size(self, input_size, resolved=None) -> Tuple[int, int]:
        return super().resolve_input_size(input_size, resolved)

    # -- ternaus native path ----------------------------------------------
    def build_native(self, resolved: ResolvedWeights, device: str):
        try:
            from retinaface.pre_trained_models import get_model
        except Exception as e:  # pragma: no cover
            raise RuntimeUnavailableError(
                "ternaus weights need the 'retinaface-pytorch' package (included in the [torch] extra): "
                "pip install 'online-face-detection[torch]'"
            ) from e
        variant = resolved.meta.get("variant", "resnet50_2020-07-20")
        dev = "cpu" if str(device).startswith("mps") else device  # ternaus pipeline is unreliable on MPS
        if dev != device:
            _log.info("ternaus runs on CPU on this machine (MPS unsupported by its pipeline)")
        try:
            model = get_model(variant, max_size=1024, device=dev)
        except TypeError:
            model = get_model(variant, max_size=1024)
        model.eval()
        return model

    def native_predict(self, model, frame_bgr, params: Dict[str, Any]) -> Dict[str, Any]:
        import numpy as np

        img_rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])  # ternaus expects RGB
        anns = model.predict_jsons(
            img_rgb,
            confidence_threshold=float(params.get("conf", 0.5)),
            nms_threshold=float(params.get("nms", 0.4)),
        )
        boxes, scores, lms = [], [], []
        for a in anns:
            bb = a.get("bbox") or []
            if len(bb) != 4:
                continue
            boxes.append(bb)
            scores.append(float(a.get("score", 0.0)))
            l = a.get("landmarks") or []
            lms.append(l if len(l) == 5 else [[0.0, 0.0]] * 5)
        return {
            "boxes": np.asarray(boxes, dtype="float32").reshape(-1, 4),
            "scores": np.asarray(scores, dtype="float32").reshape(-1),
            "landmarks": np.asarray(lms, dtype="float32").reshape(-1, 5, 2),
        }
