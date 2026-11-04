"""YuNet: A Tiny Millisecond-level Face Detector — PyTorch 复现.

论文: W. Wu, H. Peng, S. Yu. "YuNet: A Tiny Millisecond-level Face Detector".
Machine Intelligence Research, vol.20, no.5, pp.656-665, 2023.
DOI: 10.1007/s11633-023-1423-y
"""

from .model import build_yunet, YuNet, ConvDPUnit, ConvHead, DWBlock, YuNetBackbone, TFPN, YuNetHead
from .engine import (
    YuNetCriterion,
    YuNetPostprocessor,
    SimOTAAssigner,
    MlvlPointGenerator,
    eiou_loss,
    bbox_overlaps,
    bbox_decode,
    kps_decode,
    batched_nms,
)
from .dataset import WIDERFaceDataset, build_train_transforms, build_eval_transforms

__all__ = [
    "build_yunet", "YuNet", "ConvDPUnit", "ConvHead", "DWBlock",
    "YuNetBackbone", "TFPN", "YuNetHead",
    "YuNetCriterion", "YuNetPostprocessor", "SimOTAAssigner",
    "MlvlPointGenerator", "eiou_loss", "bbox_overlaps",
    "bbox_decode", "kps_decode", "batched_nms",
    "WIDERFaceDataset", "build_train_transforms", "build_eval_transforms",
]
