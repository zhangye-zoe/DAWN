from __future__ import annotations
from collections import OrderedDict
import torch
from torch import nn
from .net_utils import DenseBlock, Net, ResidualBlock, TFSamepaddingLayer, UpSample2x
from dawn.utils.tensor import crop_op, crop_to_shape


class HoVerNet(Net):
    """HoVer-Net segmentation branch used by DAWN.

    Outputs:
        np: 2-channel nuclei/background logits.
        hv: 2-channel horizontal/vertical regression map.
        seg_encoding: compact encoder representation used for CFC.
    """

    def __init__(self, input_ch: int = 3, nr_types: int | None = None, freeze: bool = False, mode: str = "fast", encoding_dim: int = 32):
        super().__init__()
        assert mode in {"original", "fast"}
        self.mode = mode
        self.freeze = freeze
        self.nr_types = nr_types
        self.output_ch = 3 if nr_types is None else 4

        module_list = [
            ("/", nn.Conv2d(input_ch, 64, 7, stride=1, padding=0, bias=False)),
            ("bn", nn.BatchNorm2d(64, eps=1e-5)),
            ("relu", nn.ReLU(inplace=True)),
        ]
        if mode == "fast":
            module_list = [("pad", TFSamepaddingLayer(ksize=7, stride=1))] + module_list
        self.conv0 = nn.Sequential(OrderedDict(module_list))
        self.d0 = ResidualBlock(64, [1, 3, 1], [64, 64, 256], 3, stride=1)
        self.d1 = ResidualBlock(256, [1, 3, 1], [128, 128, 512], 4, stride=2)
        self.d2 = ResidualBlock(512, [1, 3, 1], [256, 256, 1024], 6, stride=2)
        self.d3 = ResidualBlock(1024, [1, 3, 1], [512, 512, 2048], 3, stride=2)
        self.conv_bot = nn.Conv2d(2048, 1024, 1, stride=1, padding=0, bias=False)
        self.embedding = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(2048, 128), nn.ReLU(inplace=True), nn.Linear(128, encoding_dim))

        def branch(out_ch: int, ksize: int):
            return nn.Sequential(OrderedDict([
                ("u3", nn.Sequential(OrderedDict([
                    ("conva", nn.Conv2d(1024, 256, ksize, stride=1, padding=0, bias=False)),
                    ("dense", DenseBlock(256, [1, ksize], [128, 32], 8, split=4)),
                    ("convf", nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False)),
                ]))),
                ("u2", nn.Sequential(OrderedDict([
                    ("conva", nn.Conv2d(512, 128, ksize, stride=1, padding=0, bias=False)),
                    ("dense", DenseBlock(128, [1, ksize], [128, 32], 4, split=4)),
                    ("convf", nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False)),
                ]))),
                ("u1", nn.Sequential(OrderedDict([
                    ("conva/pad", TFSamepaddingLayer(ksize=ksize, stride=1)),
                    ("conva", nn.Conv2d(256, 64, ksize, stride=1, padding=0, bias=False)),
                ]))),
                ("u0", nn.Sequential(OrderedDict([
                    ("bn", nn.BatchNorm2d(64, eps=1e-5)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("conv", nn.Conv2d(64, out_ch, 1, stride=1, padding=0, bias=True)),
                ]))),
            ]))

        ksize = 5 if mode == "original" else 3
        items = [("np", branch(2, ksize)), ("hv", branch(2, ksize))]
        if nr_types is not None:
            items.insert(0, ("tp", branch(nr_types, ksize)))
        self.decoder = nn.ModuleDict(OrderedDict(items))
        self.upsample2x = UpSample2x()
        self.weights_init()

    def forward(self, imgs: torch.Tensor) -> dict[str, torch.Tensor]:
        imgs = imgs.float()
        if imgs.max() > 2:
            imgs = imgs / 255.0
        d0 = self.conv0(imgs)
        d0 = self.d0(d0, self.freeze) if self.training else self.d0(d0)
        with torch.set_grad_enabled(not self.freeze):
            d1 = self.d1(d0)
            d2 = self.d2(d1)
            d3 = self.d3(d2)
        embed = self.embedding(d3)
        d3 = self.conv_bot(d3)
        skips = [d0, d1, d2, d3]
        out = {}
        for name, decoder in self.decoder.items():
            up3 = self.upsample2x(d3)
            u3 = up3 + crop_to_shape(skips[2], up3)
            u3 = decoder.u3(u3)

            up2 = self.upsample2x(u3)
            u2 = up2 + crop_to_shape(skips[1], up2)
            u2 = decoder.u2(u2)

            up1 = self.upsample2x(u2)
            u1 = up1 + crop_to_shape(skips[0], up1)
            u1 = decoder.u1(u1)

            u0 = decoder.u0(u1)
            out[name] = u0
        out["seg_encoding"] = embed
        return out
