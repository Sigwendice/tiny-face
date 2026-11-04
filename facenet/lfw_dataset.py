# -*- coding: utf-8 -*-
"""LFW 数据集: 按身份文件夹构建标签, 按身份划分 train/val, 并生成验证 pairs"""
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXTS = {".jpg", ".jpeg", ".png"}

TRAIN_TF = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

EVAL_TF = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def scan_lfw(data_dir, min_images=10):
    """扫描身份文件夹, 返回 {id: [img_path, ...]} (仅保留 >= min_images 张的身份)"""
    data_dir = Path(data_dir)
    by_id = defaultdict(list)
    for person_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        imgs = [p for p in person_dir.iterdir()
                if p.suffix.lower() in IMG_EXTS]
        if len(imgs) >= min_images:
            by_id[person_dir.name] = sorted(imgs)
    return dict(by_id)


def split_ids(by_id, val_ratio=0.15, seed=42):
    """按身份划分 (同一人不会同时出现在 train 和 val)"""
    ids = sorted(by_id.keys())
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(2, int(len(ids) * val_ratio))
    val_ids = set(ids[:n_val])
    train = {k: v for k, v in by_id.items() if k not in val_ids}
    val = {k: v for k, v in by_id.items() if k in val_ids}
    return train, val


class LFWDataset(Dataset):
    def __init__(self, by_id, transform=TRAIN_TF):
        self.samples = []
        for label, name in enumerate(sorted(by_id.keys())):
            for path in by_id[name]:
                self.samples.append((str(path), label))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def make_pairs(by_id, n_pos=500, n_neg=500, seed=42):
    """生成验证 pairs: [(path1, path2, is_same), ...]"""
    rng = random.Random(seed)
    names = sorted(by_id.keys())
    pairs = []
    for _ in range(n_pos):
        name = rng.choice(names)
        a, b = rng.sample(by_id[name], 2)
        pairs.append((str(a), str(b), 1))
    for _ in range(n_neg):
        n1, n2 = rng.sample(names, 2)
        a = rng.choice(by_id[n1])
        b = rng.choice(by_id[n2])
        pairs.append((str(a), str(b), 0))
    rng.shuffle(pairs)
    return pairs


def dataset_stats(by_id):
    n_imgs = sum(len(v) for v in by_id.values())
    cnt = Counter(len(v) for v in by_id.values())
    top = ", ".join(f"{k}张x{cnt[k]}人" for k in sorted(cnt)[:3])
    return f"{len(by_id)} 个身份 / {n_imgs} 张图 (每人图片数分布: {top}...)"
