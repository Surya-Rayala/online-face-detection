"""Prior boxes and SSD-style decode for RetinaFace.

Priors are generated once per input size (cached) in the exact order the
detection heads emit anchors (cell-major, anchor-minor), so decode lines up
with the raw outputs from any runtime. Decode runs on whatever device the raw
tensors live on, keeping the GPU path copy-free.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Sequence, Tuple

import numpy as np

# steps / min_sizes are identical for mobilenet0.25 and resnet50.
_STEPS = (8, 16, 32)
_MIN_SIZES = ((16, 32), (64, 128), (256, 512))


@lru_cache(maxsize=8)
def _priors_np(h: int, w: int) -> np.ndarray:
    blocks = []
    for k, step in enumerate(_STEPS):
        fh, fw = math.ceil(h / step), math.ceil(w / step)
        mss = _MIN_SIZES[k]
        ys = (np.arange(fh) + 0.5) * step / h
        xs = (np.arange(fw) + 0.5) * step / w
        gy, gx = np.meshgrid(ys, xs, indexing="ij")
        gx = gx.reshape(-1, 1)
        gy = gy.reshape(-1, 1)
        na = len(mss)
        cx = np.repeat(gx, na, axis=1)
        cy = np.repeat(gy, na, axis=1)
        skx = np.repeat(np.array([m / w for m in mss]).reshape(1, na), gx.shape[0], axis=0)
        sky = np.repeat(np.array([m / h for m in mss]).reshape(1, na), gy.shape[0], axis=0)
        blocks.append(np.stack([cx, cy, skx, sky], axis=2).reshape(-1, 4))
    return np.concatenate(blocks, axis=0).astype("float32")


def prior_box(image_size: Tuple[int, int], device: str):
    """Return priors ``(P, 4)`` as (cx, cy, s_x, s_y), normalised to [0, 1]."""
    import torch

    arr = _priors_np(int(image_size[0]), int(image_size[1]))
    return torch.from_numpy(arr).to(device)


def decode(loc, priors, variances: Sequence[float]):
    """Decode box regressions -> normalised xyxy ``(P, 4)``."""
    import torch

    boxes = torch.cat(
        (
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1]),
        ),
        dim=1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landm(pre, priors, variances: Sequence[float]):
    """Decode 5-point landmark regressions -> normalised ``(P, 10)``."""
    import torch

    return torch.cat(
        [priors[:, :2] + pre[:, 2 * i:2 * i + 2] * variances[0] * priors[:, 2:] for i in range(5)],
        dim=1,
    )
