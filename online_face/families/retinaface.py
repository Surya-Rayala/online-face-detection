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
        # A graph that bakes decode+NMS emits 4 padded outputs; the raw graph emits
        # 3 (loc, conf, landms). Auto-detect so no runtime flag threading is needed.
        if len(raw) == 4:
            return self._postprocess_graph(raw, ctx)

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

    def _postprocess_graph(self, raw, ctx) -> Dict[str, Any]:
        """Graph already did decode + NMS (padded outputs). Just unpad + rescale."""
        import torch

        num, boxes, scores, lms = raw
        n = int(torch.as_tensor(num).reshape(-1)[0].item())
        boxes = torch.as_tensor(boxes).reshape(-1, 4)[:n].float()
        scores = torch.as_tensor(scores).reshape(-1)[:n].float()
        lms = torch.as_tensor(lms).reshape(-1, 10)[:n].reshape(-1, 5, 2).float()
        if n > 0:
            boxes = rescale_boxes(boxes, ctx["meta"])
            lms = rescale_points(lms, ctx["meta"])
        return {"boxes": boxes.detach().cpu().numpy(),
                "scores": scores.detach().cpu().numpy(),
                "landmarks": lms.detach().cpu().numpy()}

    def export_spec(self, input_size: Tuple[int, int], postprocess: str = "raw",
                    max_faces: int = 256) -> ExportSpec:
        if postprocess == "graph":
            # decode + NMS baked in; fixed, padded detections. opset 17 for NonMaxSuppression.
            return ExportSpec(input_names=["input"],
                              output_names=["num_detections", "boxes", "scores", "landmarks"],
                              dynamic_axes=None, opset=17,
                              postprocess_in_graph=True, variant=f"pp-graph-mf{int(max_faces)}")
        return ExportSpec(input_names=["input"], output_names=["loc", "conf", "landms"],
                          dynamic_axes=None, opset=13)

    def build_export_module(self, resolved: ResolvedWeights, input_size: Tuple[int, int],
                            conf: float, nms: float, max_faces: int = 256):
        """An nn.Module = base RetinaFace + decode + NMS + fixed padding, emitting
        (num_detections, boxes, scores, landmarks) in letterboxed-input pixel coords.
        Priors are baked for this exact input size (so the artifact is size-specific)."""
        import torch
        from torchvision.ops import nms as tv_nms

        from ..models.retinaface import decode, decode_landm

        base = self.build_module(resolved, "cpu", "fp32").eval()
        h_in, w_in = int(input_size[0]), int(input_size[1])
        priors = prior_box((h_in, w_in), "cpu")
        v = _VARIANCE
        conf_t, iou_t, mf = float(conf), float(nms), int(max_faces)

        class _RetinaWithPostprocess(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base = base
                self.register_buffer("priors", priors)
                self.register_buffer("box_scale", torch.tensor([w_in, h_in, w_in, h_in], dtype=torch.float32))
                self.register_buffer("lm_scale", torch.tensor([w_in, h_in] * 5, dtype=torch.float32))

            def forward(self, x):
                loc, conf, landms = self.base(x)
                boxes = decode(loc[0].float(), self.priors, v) * self.box_scale
                lms = decode_landm(landms[0].float(), self.priors, v) * self.lm_scale
                scores = conf[0].float()[:, 1]
                keep = scores > conf_t
                boxes, scores, lms = boxes[keep], scores[keep], lms[keep]
                order = tv_nms(boxes, scores, iou_t)[:mf]
                boxes, scores, lms = boxes[order], scores[order], lms[order]
                n = scores.shape[0]
                # pad to a fixed mf rows (Concat + Slice — exports cleanly, no dynamic pad size)
                ob = torch.cat([boxes, boxes.new_zeros((mf, 4))], 0)[:mf].unsqueeze(0)
                os_ = torch.cat([scores, scores.new_zeros((mf,))], 0)[:mf].unsqueeze(0)
                ol = torch.cat([lms, lms.new_zeros((mf, 10))], 0)[:mf].unsqueeze(0)
                num = torch.zeros((1,), dtype=torch.int64) + n
                return num, ob, os_, ol

        return _RetinaWithPostprocess().eval()

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
