from __future__ import annotations
from pathlib import Path
import torch


def clean_state_dict(state: dict) -> dict:
    return {k.replace("module.", ""): v for k, v in state.items()}


def load_checkpoint(model, path: str | Path, strict: bool = False):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    missing, unexpected = model.load_state_dict(clean_state_dict(state), strict=strict)
    return {"missing": missing, "unexpected": unexpected, "epoch": ckpt.get("epoch")}


def save_checkpoint(path: str | Path, model, optimizer=None, epoch: int = 0, extra: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"epoch": epoch, "state_dict": model.state_dict()}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)
