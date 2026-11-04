"""YuNet 网络结构（论文 Sec. 3.2 / Fig. 4）.

三部分组成:
  - Backbone: ConvHead(stage0) + 5 个 DWBlock（深度可分离卷积），在 1/8, 1/16, 1/32 输出三级特征
  - Neck:     TFPN —— 用 DWUnit 替换 FPN 中的标准卷积，去掉 1x1 通道对齐层
  - Head:     anchor-free 解耦头（cls/obj/bbox/kps 四个分支），stride 8/16/32 各一份

模块命名与官方 libfacedetection.train 保持一致，官方 .pth 权重可直接 strict=True 加载。

复现校验点（论文 Table 1/2）:
  yunet_n: 75,856 params / 148.7 MFLOPs @320x320
  yunet_s: 54,608 params /  96.4 MFLOPs @320x320
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 基础模块（论文 Fig.4 右上角标注）
#   DWUnit  = Conv1x1(g=1) -> Conv3x3(g=out) -> BN -> ReLU
#   DWBlock = DWUnit + DWUnit
# 注意: 与 MobileNet 相反, 这里是先 pointwise 再 depthwise
# ---------------------------------------------------------------------------
class ConvDPUnit(nn.Module):
    """DWUnit: 1x1 pointwise + 3x3 depthwise (+BN+ReLU)."""

    def __init__(self, in_ch: int, out_ch: int, with_bn_relu: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=True, groups=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=True, groups=out_ch)
        self.with_bn_relu = with_bn_relu
        if with_bn_relu:
            self.bn = nn.BatchNorm2d(out_ch)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        if self.with_bn_relu:
            x = self.relu(self.bn(x))
        return x


class ConvHead(nn.Module):
    """Stage 0: Conv3x3 s2 p1 (+BN+ReLU) 后接一个 DWUnit（论文 Fig.4 的 ConvHead,16）."""

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, 2, 1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = ConvDPUnit(mid_ch, out_ch, with_bn_relu=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.bn1(self.conv1(x)))
        return self.conv2(x)


class DWBlock(nn.Module):
    """Backbone 主体: 两个 ConvDPUnit 堆叠（论文 Fig.4 的 DWBlock）."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvDPUnit(in_ch, in_ch)
        self.conv2 = ConvDPUnit(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


# ---------------------------------------------------------------------------
# Backbone（论文 Sec. 3.2.1）: stage0(1/2) + maxpool 后 5 个 DWBlock
#   yunet_n: (3,16,16) -> (16,64) -> (64,64)x4, 在第 4/5/6 个模块后输出 1/8,1/16,1/32
# ---------------------------------------------------------------------------
class YuNetBackbone(nn.Module):
    def __init__(
        self,
        stage_channels: tuple[tuple[int, ...], ...],
        downsample_idx: tuple[int, ...] = (0, 2, 3, 4),
        out_idx: tuple[int, ...] = (3, 4, 5),
    ):
        super().__init__()
        self.layer_num = len(stage_channels)
        self.downsample_idx = downsample_idx
        self.out_idx = out_idx
        self.model0 = ConvHead(*stage_channels[0])
        for i in range(1, self.layer_num):
            self.add_module(f"model{i}", DWBlock(*stage_channels[i]))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        out = []
        for i in range(self.layer_num):
            x = getattr(self, f"model{i}")(x)
            if i in self.out_idx:
                out.append(x)
            if i in self.downsample_idx:
                x = F.max_pool2d(x, 2)
        return out


# ---------------------------------------------------------------------------
# TFPN neck（论文 Sec. 3.2.2 / Fig. 5b）
#   P_i = F_i( C_i + up(P_{i+1}) ), F_i 用 DWUnit, 全部 64 通道故无需 1x1 对齐
# ---------------------------------------------------------------------------
class TFPN(nn.Module):
    def __init__(self, in_channels: tuple[int, ...] = (64, 64, 64),
                 out_idx: tuple[int, ...] = (0, 1, 2)):
        super().__init__()
        self.out_idx = out_idx
        self.lateral_convs = nn.ModuleList(
            ConvDPUnit(c, c) for c in in_channels
        )

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        feats = list(feats)
        n = len(feats)
        for i in range(n - 1, 0, -1):
            feats[i] = self.lateral_convs[i](feats[i])
            feats[i - 1] = feats[i - 1] + F.interpolate(
                feats[i], scale_factor=2.0, mode="nearest")
        feats[0] = self.lateral_convs[0](feats[0])
        return [feats[i] for i in self.out_idx]


# ---------------------------------------------------------------------------
# Head（论文 Sec. 3.2.3）: anchor-free 解耦头
#   每级: 1 个共享 DWUnit + cls(1)/obj(1)/bbox(4)/kps(10) 四个无 BN 分支
# ---------------------------------------------------------------------------
class YuNetHead(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        in_channels: int = 64,
        feat_channels: int = 64,
        shared_stacked_convs: int = 1,
        strides: tuple[int, ...] = (8, 16, 32),
        kps_num: int = 5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.kps_num = kps_num
        self.shared_stacked_convs = shared_stacked_convs

        self.multi_level_share_convs = nn.ModuleList()
        self.multi_level_cls = nn.ModuleList()
        self.multi_level_bbox = nn.ModuleList()
        self.multi_level_obj = nn.ModuleList()
        self.multi_level_kps = nn.ModuleList()

        for _ in strides:
            if shared_stacked_convs > 0:
                self.multi_level_share_convs.append(
                    nn.Sequential(
                        *[ConvDPUnit(in_channels if i == 0 else feat_channels,
                                     feat_channels)
                          for i in range(shared_stacked_convs)]))
            ch = in_channels if shared_stacked_convs == 0 else feat_channels
            self.multi_level_cls.append(ConvDPUnit(ch, num_classes, False))
            self.multi_level_bbox.append(ConvDPUnit(ch, 4, False))
            self.multi_level_obj.append(ConvDPUnit(ch, 1, False))
            self.multi_level_kps.append(ConvDPUnit(ch, kps_num * 2, False))

    def forward(self, feats: list[torch.Tensor]):
        if self.shared_stacked_convs > 0:
            feats = [conv(f) for f, conv in zip(feats, self.multi_level_share_convs)]
        cls_preds = [conv(f) for f, conv in zip(feats, self.multi_level_cls)]
        bbox_preds = [conv(f) for f, conv in zip(feats, self.multi_level_bbox)]
        obj_preds = [conv(f) for f, conv in zip(feats, self.multi_level_obj)]
        kps_preds = [conv(f) for f, conv in zip(feats, self.multi_level_kps)]
        return cls_preds, bbox_preds, obj_preds, kps_preds


# ---------------------------------------------------------------------------
# 整网
# ---------------------------------------------------------------------------
class YuNet(nn.Module):
    def __init__(self, variant: str = "yunet_n"):
        super().__init__()
        if variant == "yunet_n":
            # 论文 Table 2: YuNet 行 —— Stage1 64 通道, head 两个 DWUnit
            stage_channels = ((3, 16, 16), (16, 64), (64, 64), (64, 64), (64, 64), (64, 64))
            shared_stacked_convs = 1
        elif variant == "yunet_s":
            # 论文 Table 2: YuNet-s 行 —— Stage1 半通道(16->32), head 单 DWUnit
            stage_channels = ((3, 16, 16), (16, 32), (32, 64), (64, 64), (64, 64), (64, 64))
            shared_stacked_convs = 0
        else:
            raise ValueError(f"unknown variant: {variant}")

        self.variant = variant
        self.backbone = YuNetBackbone(stage_channels,
                                      downsample_idx=(0, 2, 3, 4),
                                      out_idx=(3, 4, 5))
        self.neck = TFPN()
        self.bbox_head = YuNetHead(shared_stacked_convs=shared_stacked_convs)

    def extract_feat(self, img: torch.Tensor) -> list[torch.Tensor]:
        return self.neck(self.backbone(img))

    def forward(self, img: torch.Tensor):
        feats = self.extract_feat(img)
        return self.bbox_head(feats)


def build_yunet(variant: str = "yunet_n") -> YuNet:
    return YuNet(variant)
