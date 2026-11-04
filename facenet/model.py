# -*- coding: utf-8 -*-
"""MobileFaceNet backbone (112x112 input, 512-d embedding, ~1M params).

标准结构: 3x3 conv s2 -> dw conv -> 4 个 stage 的 DepthWise bottleneck + Residual,
最后 1x1 conv 投影到 512 维 embedding。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """conv3x3/1x1 + BN + PReLU"""

    def __init__(self, in_c, out_c, k, s, p, dw=False, linear=False):
        super().__init__()
        if dw:
            self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=in_c, bias=False)
        else:
            self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.Identity() if linear else nn.PReLU(out_c)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DepthWise(nn.Module):
    """1x1 expand (t倍) -> dw conv3x3 stride s -> 1x1 project, 可选残差"""

    def __init__(self, in_c, out_c, s=1, residual=False):
        super().__init__()
        self.residual = residual and (s == 1) and (in_c == out_c)
        self.expand = ConvBlock(in_c, in_c * 2, 1, 1, 0)
        self.dw = ConvBlock(in_c * 2, in_c * 2, 3, s, 1, dw=True)
        self.project = ConvBlock(in_c * 2, out_c, 1, 1, 0, linear=True)

    def forward(self, x):
        out = self.project(self.dw(self.expand(x)))
        if self.residual:
            out = out + x
        return out


class Residual(nn.Module):
    """堆叠 n 个 stride=1 的 DepthWise 残差块"""

    def __init__(self, c, num_block):
        super().__init__()
        self.blocks = nn.ModuleList(
            [DepthWise(c, c, s=1, residual=True) for _ in range(num_block)]
        )

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


class MobileFaceNet(nn.Module):
    def __init__(self, embedding_size=512, dropout=0.5):
        super().__init__()
        self.conv1 = ConvBlock(3, 64, 3, 2, 1)          # 112 -> 56
        self.dw1 = ConvBlock(64, 64, 3, 1, 1, dw=True)  # 56 -> 56
        self.stage1 = DepthWise(64, 64, s=2)            # 56 -> 28
        self.res1 = Residual(64, 5)
        self.stage2 = DepthWise(64, 128, s=2)           # 28 -> 14
        self.res2 = Residual(128, 1)
        self.stage3 = DepthWise(128, 128, s=2)          # 14 -> 7
        self.res3 = Residual(128, 6)
        self.stage4 = DepthWise(128, 128, s=2)          # 7 -> 4
        self.res4 = Residual(128, 2)
        self.stage5 = DepthWise(128, 128, s=2)          # 4 -> 2
        self.proj = ConvBlock(128, 512, 2, 1, 0, linear=True)  # 2 -> 1
        self.bn = nn.BatchNorm1d(embedding_size)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_size, embedding_size, bias=False)
        self.bn_out = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.dw1(self.conv1(x))
        x = self.res1(self.stage1(x))
        x = self.res2(self.stage2(x))
        x = self.res3(self.stage3(x))
        x = self.res4(self.stage4(x))
        x = self.proj(self.stage5(x))
        x = x.flatten(1)
        x = self.bn(x)
        x = self.drop(x)
        x = self.bn_out(self.fc(x))
        return F.normalize(x, dim=1)  # L2 归一化 embedding


if __name__ == "__main__":
    m = MobileFaceNet()
    n_params = sum(p.numel() for p in m.parameters())
    y = m(torch.randn(2, 3, 112, 112))
    print(f"params: {n_params / 1e6:.2f}M, output: {tuple(y.shape)}")
