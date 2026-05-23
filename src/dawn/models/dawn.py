from __future__ import annotations
import torch
from torch import nn
from .detector import ResUNetDetector
from .hovernet import HoVerNet


class DAWN(nn.Module):
    def __init__(self, detector_backbone: str = "resnet34", detector_pretrained: bool = True, hover_mode: str = "fast", encoding_dim: int = 32):
        super().__init__()
        self.detector = ResUNetDetector(detector_backbone, out_channels=1, pretrained=detector_pretrained, encoding_dim=encoding_dim)
        self.segmentor = HoVerNet(mode=hover_mode, encoding_dim=encoding_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.segmentor(x)
        out.update(self.detector(x))
        return out

    def load_segmentor(self, checkpoint_path: str, strict: bool = False) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt.get("model", ckpt))
        state = {k.replace("module.", "").replace("segmentor.", ""): v for k, v in state.items() if not k.startswith("detector.")}
        self.segmentor.load_state_dict(state, strict=strict)
