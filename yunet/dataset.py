"""WIDER FACE (labelv2 格式, 来自 SCRFD) 数据集 + 训练数据增强.

论文 Sec. 4.1: 尺度增强是关键训练策略 —— 以短边 × scale∈[0.5,1.5] 裁方形 patch,
越界区域填充均值色(128), 再 resize 到训练尺寸; 另有随机水平翻转 p=0.5。
论文 Sec. 4.2 / Table 3: scale 分布与原始训练集一致时([0.5,1.5])平均 AP 最优。

labelv2 格式:
  # <relative_path> <width> <height>
  <x1> <y1> <x2> <y2> <x1> <y1> <v1> ... (5 点关键点 x/y/v 交错) <attrs...>
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# 标注解析
# ---------------------------------------------------------------------------
def parse_labelv2(ann_file: str) -> list[dict]:
    records = []
    with open(ann_file, "r", encoding="utf-8") as f:
        cur = None
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                parts = line[1:].split()
                cur = dict(path=parts[0], width=int(parts[1]), height=int(parts[2]),
                           boxes=[], keypoints=[], ignored=[])
                records.append(cur)
                continue
            vals = [float(v) for v in line.split()]
            bbox = np.array(vals[0:4], dtype=np.float32)
            kps = np.zeros((5, 3), dtype=np.float32)
            ignore = False
            if len(vals) > 5:
                kps = np.array(vals[4:19], dtype=np.float32).reshape(5, 3)
                for p in kps:
                    p[2] = 0.0 if (p == -1).all() else 1.0
            elif len(vals) == 5 and vals[4] == 1:
                ignore = True  # 仅 4+1 值: ignore 标记
            (cur["ignored"] if ignore else cur["boxes"]).append(bbox)
            if not ignore:
                cur["keypoints"].append(kps)
    # 修正形状
    for r in records:
        r["boxes"] = np.array(r["boxes"], dtype=np.float32).reshape(-1, 4)
        r["keypoints"] = np.array(r["keypoints"], dtype=np.float32).reshape(-1, 5, 3)
        r["ignored"] = np.array(r["ignored"], dtype=np.float32).reshape(-1, 4)
        r["labels"] = np.zeros(len(r["boxes"]), dtype=np.int64)
    return records


# ---------------------------------------------------------------------------
# 数据增强（论文 Sec. 4.1）
# ---------------------------------------------------------------------------
class RandomSquareCrop:
    """短边 × scale 裁方形 patch, 越界填 128; 至少保留一个中心在 patch 内的框."""

    def __init__(self, crop_choice: Sequence[float] = (0.5, 0.7, 0.9, 1.1, 1.3, 1.5)):
        self.crop_choice = tuple(crop_choice)

    def __call__(self, sample: dict) -> dict:
        image, boxes = sample["image"], sample["boxes"]
        if boxes.shape[0] == 0:
            return sample
        h, w = image.shape[:2]
        max_scale = max(self.crop_choice)
        scale, retry = None, 0
        while True:
            retry += 1
            if retry == 1 or max_scale > 1.0:
                scale = float(np.random.choice(self.crop_choice))
            else:
                scale *= 1.2
            crop = int(scale * min(w, h))
            for _ in range(250):
                left = 0 if crop == w else np.random.randint(min(0, w - crop), max(0, w - crop))
                top = 0 if crop == h else np.random.randint(min(0, h - crop), max(0, h - crop))
                patch = np.array([left, top, left + crop, top + crop], dtype=np.int64)
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2
                keep = ((centers[:, 0] > patch[0]) & (centers[:, 1] > patch[1])
                        & (centers[:, 0] < patch[2]) & (centers[:, 1] < patch[3]))
                if not keep.any():
                    continue
                _crop_sample(sample, patch, keep)
                return sample


class Resize:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, sample: dict) -> dict:
        image = sample["image"]
        h, w = image.shape[:2]
        sw, sh = self.size / w, self.size / h
        sample["image"] = cv2.resize(image, (self.size, self.size))
        sample["boxes"][:, 0::2] *= sw
        sample["boxes"][:, 1::2] *= sh
        sample["boxes"][:, 0::2] = sample["boxes"][:, 0::2].clip(0, self.size)
        sample["boxes"][:, 1::2] = sample["boxes"][:, 1::2].clip(0, self.size)
        if len(sample["keypoints"]):
            sample["keypoints"][:, :, 0] = (sample["keypoints"][:, :, 0] * sw).clip(0, self.size)
            sample["keypoints"][:, :, 1] = (sample["keypoints"][:, :, 1] * sh).clip(0, self.size)
        if len(sample["ignored"]):
            sample["ignored"][:, 0::2] *= sw
            sample["ignored"][:, 1::2] *= sh
        return sample


class FilterSmallBoxes:
    """过滤过小正样本(增强后无可辨认特征的 tiny face)."""

    def __init__(self, min_size: float = 10.0):
        self.min_size = min_size

    def __call__(self, sample: dict) -> dict:
        boxes = sample["boxes"]
        if boxes.shape[0]:
            keep = ((boxes[:, 2] - boxes[:, 0]) >= self.min_size) \
                & ((boxes[:, 3] - boxes[:, 1]) >= self.min_size)
            sample["boxes"] = boxes[keep]
            sample["keypoints"] = sample["keypoints"][keep]
            sample["labels"] = sample["labels"][keep]
        return sample


class RandomHorizontalFlip:
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, sample: dict) -> dict:
        if np.random.random() >= self.prob:
            return sample
        w = sample["image"].shape[1]
        sample["image"] = np.flip(sample["image"], axis=1).copy()
        b = sample["boxes"]
        if b.shape[0]:
            x1 = w - b[:, 2].copy()
            x2 = w - b[:, 0].copy()
            b[:, 0], b[:, 2] = x1, x2
            kps = sample["keypoints"][:, [1, 0, 2, 4, 3], :].copy()
            kps[..., 0] = w - kps[..., 0]
            sample["keypoints"] = kps
        return sample


class ToTensor:
    def __call__(self, sample: dict) -> dict:
        img = np.ascontiguousarray(sample["image"].transpose(2, 0, 1))
        sample["image"] = torch.from_numpy(img).float()   # 0-255, 不做归一化(与官方一致)
        sample["boxes"] = torch.from_numpy(np.ascontiguousarray(sample["boxes"])).float()
        sample["labels"] = torch.from_numpy(np.ascontiguousarray(sample["labels"])).long()
        sample["keypoints"] = torch.from_numpy(np.ascontiguousarray(sample["keypoints"])).float()
        sample["ignored"] = torch.from_numpy(np.ascontiguousarray(sample["ignored"])).float()
        return sample


def build_train_transforms(image_size: int = 640,
                           crop_choice=(0.5, 0.7, 0.9, 1.1, 1.3, 1.5),
                           min_box_size: float = 10.0):
    return (RandomSquareCrop(crop_choice), Resize(image_size),
            FilterSmallBoxes(min_box_size), RandomHorizontalFlip(0.5), ToTensor())


def build_eval_transforms(image_size: int | None = 640):
    """推理预处理: 指定尺寸时 keep-ratio resize + 右下 padding; None 时 pad 到 32 倍数."""

    class _Eval:
        def __call__(self, sample: dict) -> dict:
            image = sample["image"]
            h, w = image.shape[:2]
            if image_size is not None:
                scale = min(image_size / w, image_size / h)
                nw, nh = int(w * scale + 0.5), int(h * scale + 0.5)
                resized = cv2.resize(image, (nw, nh))
                sample["scale"] = scale
            else:
                nw, nh = w, h
                sample["scale"] = 1.0
            if image_size is not None:
                # letterbox 到固定方形 (批处理要求等尺寸); 右下 padding 无坐标偏移
                dh = image_size - nh
                dw = image_size - nw
            else:
                dh = (int(np.ceil(nh / 32)) * 32) - nh
                dw = (int(np.ceil(nw / 32)) * 32) - nw
            padded = cv2.copyMakeBorder(resized, 0, dh, 0, dw,
                                        cv2.BORDER_CONSTANT, value=0)
            sample["image"] = torch.from_numpy(
                np.ascontiguousarray(padded.transpose(2, 0, 1))).float()
            sample["img_shape"] = (nh, nw)
            # GT 同步缩放 (右下 padding 无偏移, 直接乘 scale), 与预测同一坐标系
            for key, idx in (("boxes", None), ("ignored", None)):
                if len(sample[key]):
                    sample[key][:, 0::2] *= scale
                    sample[key][:, 1::2] *= scale
            if len(sample["keypoints"]):
                sample["keypoints"][:, :, 0] *= scale
                sample["keypoints"][:, :, 1] *= scale
            return sample

    return _Eval()


def _crop_sample(sample: dict, patch: np.ndarray, keep: np.ndarray) -> None:
    image = sample["image"]
    h, w = image.shape[:2]
    crop_h = patch[3] - patch[1]
    crop_w = patch[2] - patch[0]
    cropped = np.ones((crop_h, crop_w, image.shape[2]), dtype=image.dtype) * 128
    pf = patch.copy()
    pf[0], pf[1] = max(0, pf[0]), max(0, pf[1])
    pf[2], pf[3] = min(w, pf[2]), min(h, pf[3])
    pt = pf.copy()
    pt[0], pt[1] = max(0, -patch[0]), max(0, -patch[1])
    pt[2], pt[3] = pt[0] + pf[2] - pf[0], pt[1] + pf[3] - pf[1]
    cropped[pt[1]:pt[3], pt[0]:pt[2]] = image[pf[1]:pf[3], pf[0]:pf[2]]

    def crop_boxes(arr: np.ndarray) -> np.ndarray:
        if not len(arr):
            return arr.reshape(-1, 4)
        arr = arr[keep].astype(np.float32).copy()
        arr[:, 2:] = arr[:, 2:].clip(max=patch[2:])
        arr[:, :2] = arr[:, :2].clip(min=patch[:2])
        arr -= np.tile(patch[:2], 2)
        return arr

    def crop_kps(kps: np.ndarray) -> np.ndarray:
        if not len(kps):
            return kps.reshape(-1, 5, 3)
        kps = kps[keep].astype(np.float32).copy()
        kps[:, :, :2] = kps[:, :, :2].clip(max=patch[2:])
        kps[:, :, :2] = kps[:, :, :2].clip(min=patch[:2])
        kps[:, :, 0] -= patch[0]
        kps[:, :, 1] -= patch[1]
        return kps

    def crop_ignored(arr: np.ndarray) -> np.ndarray:
        """ignored 框独立按中心落在 patch 内过滤 (数量与 boxes 不同, 不能复用 keep)"""
        if not len(arr):
            return arr.reshape(-1, 4)
        arr = arr.astype(np.float32).copy()
        centers = (arr[:, :2] + arr[:, 2:]) / 2
        k = ((centers[:, 0] > patch[0]) & (centers[:, 1] > patch[1])
             & (centers[:, 0] < patch[2]) & (centers[:, 1] < patch[3]))
        arr = arr[k]
        arr[:, 2:] = arr[:, 2:].clip(max=patch[2:])
        arr[:, :2] = arr[:, :2].clip(min=patch[:2])
        arr -= np.tile(patch[:2], 2)
        return arr

    sample["image"] = cropped
    sample["boxes"] = crop_boxes(sample["boxes"])
    sample["keypoints"] = crop_kps(sample["keypoints"])
    sample["labels"] = sample["labels"][keep]
    sample["ignored"] = crop_ignored(sample["ignored"])


def collate(samples: list[dict]) -> dict:
    return dict(
        images=torch.stack([s["image"] for s in samples]),
        boxes=[s["boxes"] for s in samples],
        labels=[s["labels"] for s in samples],
        keypoints=[s["keypoints"] for s in samples],
        ignored=[s["ignored"] for s in samples],
        paths=[s["path"] for s in samples],
    )


class WIDERFaceDataset(Dataset):
    def __init__(self, ann_file: str, img_prefix: str, train: bool = True,
                 image_size: int = 640):
        self.records = parse_labelv2(ann_file)
        self.img_prefix = img_prefix
        self.train = train
        if train:
            self.transforms = build_train_transforms(image_size)
        else:
            self.transforms = (build_eval_transforms(image_size),)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        img = cv2.imread(os.path.join(self.img_prefix, r["path"]), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(r["path"])
        sample = dict(image=img, path=r["path"],
                      boxes=r["boxes"].copy(), labels=r["labels"].copy(),
                      keypoints=r["keypoints"].copy(), ignored=r["ignored"].copy())
        for t in self.transforms:
            sample = t(sample)
        return sample
