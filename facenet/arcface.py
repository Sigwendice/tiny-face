# -*- coding: utf-8 -*-
"""ArcFace 损失头 (InsightFace 数值稳定实现)"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFace(nn.Module):
    def __init__(self, embedding_size=512, num_classes=100, s=64.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        # cos(theta + m)
        phi = cosine * math.cos(self.m) - sine * math.sin(self.m)
        # theta > pi - m 时用 cos(theta) - sin(pi-m)*m 替代, 保证单调
        th = math.cos(math.pi - self.m)
        mm = math.sin(math.pi - self.m) * self.m
        phi = torch.where(cosine > th, phi, cosine - mm)
        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).type_as(cosine)
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        return logits * self.s
