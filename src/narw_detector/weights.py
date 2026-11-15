"""A model on disk: a folder holding weights.pt (a plain PyTorch state_dict) and config.json.

config.json records the architecture width, the render geometry, the operating point tau, and
the SHA-256 of weights.pt, which is checked on load.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from narw_detector.model import DualStreamSed
from narw_detector.spectrogram import RenderSpec

WEIGHTS, CONFIG = "weights.pt", "config.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_model(folder: Path, net: DualStreamSed, width: int, spec: RenderSpec, tau: float,
               notes: dict | None = None) -> Path:
    """Write weights.pt and config.json into folder."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, folder / WEIGHTS)
    config = {"architecture": "DualStreamSed", "width": width, "render": spec.to_dict(),
              "tau": float(tau), "weights_sha256": sha256(folder / WEIGHTS), **(notes or {})}
    (folder / CONFIG).write_text(json.dumps(config, indent=2) + "\n")
    return folder


def read_config(folder: Path) -> dict:
    return json.loads((Path(folder) / CONFIG).read_text())


def write_tau(folder: Path, tau: float, notes: dict) -> None:
    """Replace the operating point in config.json (narw-threshold --write)."""
    config = read_config(folder) | {"tau": float(tau)} | notes
    (Path(folder) / CONFIG).write_text(json.dumps(config, indent=2) + "\n")


def load_model(folder: Path, device: str = "cpu") -> tuple[DualStreamSed, RenderSpec, float]:
    """(network in eval mode on device, render spec, tau) from a model folder."""
    folder = Path(folder)
    config = read_config(folder)
    if config["architecture"] != "DualStreamSed":
        raise ValueError(f"{folder / CONFIG}: unknown architecture {config['architecture']}")
    digest = sha256(folder / WEIGHTS)
    if digest != config["weights_sha256"]:
        raise ValueError(f"{folder / WEIGHTS}: SHA-256 {digest} does not match config.json's "
                         f"{config['weights_sha256']}. Is the download incomplete, or the file "
                         "from another model?")
    net = DualStreamSed(width=config["width"])
    net.load_state_dict(torch.load(folder / WEIGHTS, map_location="cpu", weights_only=True))
    return net.eval().to(device), RenderSpec(**config["render"]), float(config["tau"])


def pick_device(name: str = "auto") -> str:
    """The device for name; auto means cuda when available, else cpu. Apple's MPS is used only
    when asked for: some PyTorch versions crash running the model's GRUs on MPS (2.15 on
    macOS 14 does)."""
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"
