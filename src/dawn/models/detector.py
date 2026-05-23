from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch))
    def forward(self, x):
        identity = x if self.down is None else self.down(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + identity, inplace=True)


class ResNetEncoder(nn.Module):
    def __init__(self, layers=(3,4,6,3), channels=(64,64,128,256,512)):
        super().__init__()
        self.conv1 = nn.Conv2d(3, channels[0], 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.in_ch = channels[0]
        self.layer1 = self._make_layer(channels[1], layers[0], stride=1)
        self.layer2 = self._make_layer(channels[2], layers[1], stride=2)
        self.layer3 = self._make_layer(channels[3], layers[2], stride=2)
        self.layer4 = self._make_layer(channels[4], layers[3], stride=2)
    def _make_layer(self, out_ch, blocks, stride):
        mods = [BasicBlock(self.in_ch, out_ch, stride)]
        self.in_ch = out_ch
        for _ in range(1, blocks): mods.append(BasicBlock(self.in_ch, out_ch, 1))
        return nn.Sequential(*mods)
    def forward(self, x):
        c0 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.maxpool(c0)
        c1 = self.layer1(x); c2 = self.layer2(c1); c3 = self.layer3(c2); c4 = self.layer4(c3)
        return c0, c1, c2, c3, c4


def _encoder(backbone: str):
    backbone = backbone.lower()
    if backbone in {"resnet18", "resunet18"}: return ResNetEncoder((2,2,2,2)), [64,64,128,256,512]
    if backbone in {"resnet34", "resunet34"}: return ResNetEncoder((3,4,6,3)), [64,64,128,256,512]
    raise ValueError(f"Unsupported built-in detector backbone: {backbone}. Use resnet18/resnet34.")


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, p: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Dropout2d(p) if p > 0 else nn.Identity())
    def forward(self, x): return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = nn.Sequential(ConvBNAct(out_ch + skip_ch, out_ch, 0.1), ConvBNAct(out_ch, out_ch, 0.1))
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]: x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ResUNetDetector(nn.Module):
    """ResNet-UNet detector trained from point/Gaussian supervision.

    `pretrained` is accepted for config compatibility. This refactored archive
    does not download weights automatically, which makes server runs reproducible
    and offline-friendly. To use ImageNet weights, load a checkpoint explicitly.
    """
    def __init__(self, backbone: str = "resnet34", out_channels: int = 1, pretrained: bool = False, encoding_dim: int = 32):
        super().__init__()
        self.encoder, ch = _encoder(backbone)
        self.up4 = UpBlock(ch[4], ch[3], ch[3])
        self.up3 = UpBlock(ch[3], ch[2], ch[2])
        self.up2 = UpBlock(ch[2], ch[1], ch[1])
        self.up1 = UpBlock(ch[1], ch[0], ch[0])
        self.head = nn.ConvTranspose2d(ch[0], out_channels, 2, stride=2)
        self.embedding = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(ch[4], 128), nn.ReLU(inplace=True), nn.Linear(128, encoding_dim))
    def forward(self, x):
        c0, c1, c2, c3, c4 = self.encoder(x.float())
        y = self.up4(c4, c3); y = self.up3(y, c2); y = self.up2(y, c1); y = self.up1(y, c0); y = self.head(y)
        return {"det_logits": y, "det_prob": torch.sigmoid(y), "det_encoding": self.embedding(c4)}
