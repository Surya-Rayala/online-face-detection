"""Weights registry for the face package.

``weights`` may be a known key (auto-downloaded when a URL exists), a local
file path, or ``None`` (the family default). When a weight cannot be
auto-fetched, :class:`WeightsNotAvailableError` is raised with the exact path
to place the file and how to pass it.

Override / extend the table without editing code via
``~/.online/face_registry.json`` or ``$ONLINE_FACE_REGISTRY`` (a JSON file
merged over the built-ins).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

from .families.base import ResolvedWeights
from .runtime.errors import UnknownWeightsError, WeightsNotAvailableError
from .runtime.logging import get_logger

_log = get_logger("registry")

ARTIFACT_SUFFIXES = {".onnx", ".engine", ".plan", ".trt", ".torchscript", ".ts"}
DEFAULT_WEIGHT = "mobilenet0.25"

# biubug6 RetinaFace weights (clean graph, exportable to onnx/trt).
_BIUBUG6: Dict[str, dict] = {
    "mobilenet0.25": {
        "arch": "mobilenet0.25",
        "filename": "retinaface_mobilenet0.25.pth",
        "sha256": None,
        "url": "https://huggingface.co/py-feat/retinaface/resolve/main/mobilenet0.25_Final.pth",
        "input_size": 640,
        "exportable": True,
    },
    "resnet50": {
        "arch": "resnet50",
        "filename": "retinaface_resnet50.pth",
        "sha256": None,
        "url": None,  # no stable, arch-matched mirror -> manual placement
        "manual": (
            "Download 'Resnet50_Final.pth' from biubug6/Pytorch_Retinaface "
            "(https://github.com/biubug6/Pytorch_Retinaface — Google Drive link in the README)."
        ),
        "input_size": 840,
        "exportable": True,
    },
}

# ternaus retinaface-pytorch (torch-only convenience; runs its own pipeline).
_TERNAUS: Dict[str, dict] = {
    "ternaus_resnet50": {
        "arch": "resnet50",
        "variant": "resnet50_2020-07-20",
        "native": True,
        "exportable": False,
        "input_size": 1024,
    },
}


def _load_overrides() -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for p in (Path.home() / ".online" / "face_registry.json", Path(os.getenv("ONLINE_FACE_REGISTRY", ""))):
        try:
            if p and p.is_file():
                out.update(json.loads(p.read_text()))
        except Exception as e:  # pragma: no cover
            _log.warning("ignoring bad registry override %s: %s", p, e)
    return out


def _biubug6() -> Dict[str, dict]:
    merged = dict(_BIUBUG6)
    merged.update(_load_overrides())
    return merged


def available_weights(model: Optional[str] = None) -> List[str]:
    """All weight keys for the face family (``model`` is accepted for API parity)."""
    return list(_biubug6().keys()) + list(_TERNAUS.keys())


def fingerprint_file(path: Path, head_bytes: int = 1 << 20) -> str:
    h = hashlib.sha256()
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            h.update(f.read(head_bytes))
        h.update(str(size).encode())
    except OSError:
        return ""
    return h.hexdigest()[:16]


def _download(url: str, dest: Path, sha256: Optional[str]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    _log.info("downloading %s -> %s", url, dest)
    req = urllib.request.Request(url, headers={"User-Agent": "online-face/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    if sha256:
        got = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if got != sha256:
            tmp.unlink(missing_ok=True)
            raise WeightsNotAvailableError(f"sha256 mismatch for {url}: expected {sha256}, got {got}")
    os.replace(tmp, dest)


def resolve_weights(weights: Optional[str], cache) -> ResolvedWeights:
    """Resolve ``weights`` (key | path | None) to a local file."""
    biubug6 = _biubug6()

    if weights is None:
        weights = DEFAULT_WEIGHT

    key = str(weights)

    # ternaus convenience (no file to manage; runs natively).
    if key in _TERNAUS:
        e = _TERNAUS[key]
        return ResolvedWeights(
            key=key, path=Path(f"ternaus:{e['variant']}"), fingerprint="", exportable=False,
            is_artifact=False, arch=e["arch"], meta={"native": True, "variant": e["variant"], "source": "ternaus"},
        )

    # known biubug6 key.
    if key in biubug6:
        e = biubug6[key]
        dest = cache.weights_dir / e["filename"]
        if not dest.exists():
            if e.get("url"):
                try:
                    _download(e["url"], dest, e.get("sha256"))
                except WeightsNotAvailableError:
                    raise
                except Exception as ex:
                    raise WeightsNotAvailableError(
                        f"could not download weights '{key}': {ex}. "
                        f"Place the file at {dest} and pass weights='{dest}'."
                    ) from ex
            else:
                raise WeightsNotAvailableError(
                    f"weights '{key}' are not auto-downloadable. {e.get('manual', '')} "
                    f"Then place it at {dest} (or pass weights=<your path>)."
                )
        return ResolvedWeights(
            key=key, path=dest, fingerprint=fingerprint_file(dest), exportable=bool(e.get("exportable", True)),
            is_artifact=False, arch=e["arch"], meta={"source": "biubug6"},
        )

    # a file path.
    p = Path(key).expanduser()
    if p.exists():
        suffix = p.suffix.lower()
        is_artifact = suffix in ARTIFACT_SUFFIXES
        arch = "resnet50" if any(t in p.name.lower() for t in ("resnet", "r50", "re50")) else "mobilenet0.25"
        return ResolvedWeights(
            key=str(p), path=p, fingerprint=fingerprint_file(p), exportable=not is_artifact,
            is_artifact=is_artifact, arch=arch, meta={"source": "path"},
        )

    raise UnknownWeightsError(f"unknown weights {key!r}; available: {available_weights()} (or pass a file path)")
