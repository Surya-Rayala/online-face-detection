"""RetinaFace network (biubug6 architecture: MobileNet0.25 / ResNet50).

Vendored, minimal definition of the backbone + FPN + SSH + detection heads so
the graph is clean and exportable to ONNX/TensorRT. The forward returns raw
``(loc, conf, landmarks)`` with conf already soft-maxed, so decode/NMS in the
family work identically across every runtime.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

cfg_mnet: Dict = {
    "name": "mobilenet0.25",
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
    "in_channel": 32,
    "out_channel": 64,
    "return_layers": {"stage1": 1, "stage2": 2, "stage3": 3},
}

cfg_re50: Dict = {
    "name": "Resnet50",
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
    "in_channel": 256,
    "out_channel": 256,
    "return_layers": {"layer2": 1, "layer3": 2, "layer4": 3},
}


def conv_bn(inp, oup, stride=1, leaky=0.0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


def conv_bn_no_relu(inp, oup, stride=1):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup))


def conv_bn1X1(inp, oup, stride=1, leaky=0.0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


def conv_dw(inp, oup, stride=1, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


class MobileNetV1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 8, 2, leaky=0.1), conv_dw(8, 16, 1), conv_dw(16, 32, 2),
            conv_dw(32, 32, 1), conv_dw(32, 64, 2), conv_dw(64, 64, 1),
        )
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2), conv_dw(128, 128, 1), conv_dw(128, 128, 1),
            conv_dw(128, 128, 1), conv_dw(128, 128, 1), conv_dw(128, 128, 1),
        )
        self.stage3 = nn.Sequential(conv_dw(128, 256, 2), conv_dw(256, 256, 1))
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        x = x.view(-1, 256)
        return self.fc(x)


class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        assert out_channel % 4 == 0
        leaky = 0.1 if out_channel <= 64 else 0.0
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel // 2)
        self.conv5X5_1 = conv_bn(in_channel, out_channel // 4, leaky=leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel // 4, out_channel // 4)
        self.conv7X7_2 = conv_bn(out_channel // 4, out_channel // 4, leaky=leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel // 4, out_channel // 4)

    def forward(self, x):
        c3 = self.conv3X3(x)
        c5_1 = self.conv5X5_1(x)
        c5 = self.conv5X5_2(c5_1)
        c7_2 = self.conv7X7_2(c5_1)
        c7 = self.conv7x7_3(c7_2)
        return F.relu(torch.cat([c3, c5, c7], dim=1))


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        leaky = 0.1 if out_channels <= 64 else 0.0
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, leaky=leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, leaky=leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, leaky=leaky)
        self.merge1 = conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky=leaky)

    def forward(self, x):
        x = list(x.values())
        o1, o2, o3 = self.output1(x[0]), self.output2(x[1]), self.output3(x[2])
        up3 = F.interpolate(o3, size=[o2.size(2), o2.size(3)], mode="nearest")
        o2 = self.merge2(o2 + up3)
        up2 = F.interpolate(o2, size=[o1.size(2), o1.size(3)], mode="nearest")
        o1 = self.merge1(o1 + up2)
        return [o1, o2, o3]


def _head(module_cls, fpn_num, inchannels, anchors=2):
    return nn.ModuleList([module_cls(inchannels, anchors) for _ in range(fpn_num)])


class ClassHead(nn.Module):
    def __init__(self, inchannels, num_anchors=2):
        super().__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 2, 1, 1, 0)

    def forward(self, x):
        out = self.conv1x1(x).permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self, inchannels, num_anchors=2):
        super().__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, 1, 1, 0)

    def forward(self, x):
        out = self.conv1x1(x).permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self, inchannels, num_anchors=2):
        super().__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 10, 1, 1, 0)

    def forward(self, x):
        out = self.conv1x1(x).permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 10)


class RetinaFace(nn.Module):
    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        from torchvision.models._utils import IntermediateLayerGetter

        if cfg["name"] == "mobilenet0.25":
            backbone = MobileNetV1()
        elif cfg["name"] == "Resnet50":
            import torchvision.models as models

            backbone = models.resnet50(weights=None)
        else:  # pragma: no cover
            raise ValueError(f"unknown backbone {cfg['name']}")

        self.body = IntermediateLayerGetter(backbone, cfg["return_layers"])
        in_ch = cfg["in_channel"]
        in_channels_list = [in_ch * 2, in_ch * 4, in_ch * 8]
        out_ch = cfg["out_channel"]
        self.fpn = FPN(in_channels_list, out_ch)
        self.ssh1, self.ssh2, self.ssh3 = SSH(out_ch, out_ch), SSH(out_ch, out_ch), SSH(out_ch, out_ch)
        self.ClassHead = _head(ClassHead, 3, out_ch)
        self.BboxHead = _head(BboxHead, 3, out_ch)
        self.LandmarkHead = _head(LandmarkHead, 3, out_ch)

    def forward(self, inputs):
        out = self.body(inputs)
        fpn = self.fpn(out)
        features = [self.ssh1(fpn[0]), self.ssh2(fpn[1]), self.ssh3(fpn[2])]
        bbox = torch.cat([self.BboxHead[i](f) for i, f in enumerate(features)], dim=1)
        cls = torch.cat([self.ClassHead[i](f) for i, f in enumerate(features)], dim=1)
        ldm = torch.cat([self.LandmarkHead[i](f) for i, f in enumerate(features)], dim=1)
        return bbox, F.softmax(cls, dim=-1), ldm


def build_retinaface(arch: str):
    """Return ``(model, cfg)`` for ``arch`` in {"mobilenet0.25", "resnet50"}."""
    a = arch.lower()
    if a in ("mobilenet0.25", "mnet", "mobilenet", "mnet0.25"):
        cfg = cfg_mnet
    elif a in ("resnet50", "r50", "re50"):
        cfg = cfg_re50
    else:  # pragma: no cover
        raise ValueError(f"unknown RetinaFace arch {arch!r}")
    return RetinaFace(cfg), cfg
