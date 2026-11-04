"""YuNet 训练脚本（论文 Sec. 4.2 训练配置）.

论文配置:
  - SGD, momentum 0.9, weight decay 5e-4
  - batch 16 x 2 GPU, lr 0.001 线性 warmup 1500 iters 至 0.01
  - StepLR: 第 400 / 544 epoch 衰减 x0.1, 共 560 epoch, 无预训练
  - 数据: WIDER FACE train (labelv2), 尺度增强 crop_choice=(0.5..1.5), 输入 640x640

用法:
  python train.py --data-root data/widerface --device cuda --batch-size 16
  python train.py --data-root data/widerface --epochs 2 --smoke   # 冒烟
"""

from __future__ import annotations

import argparse
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from yunet import build_yunet, YuNetCriterion
from yunet.dataset import WIDERFaceDataset, collate


def parse_args():
    p = argparse.ArgumentParser(description="YuNet training (paper reproduction)")
    p.add_argument("--data-root", default="data/widerface",
                   help="包含 WIDER_train/images, WIDER_val/images, labelv2/ 的目录")
    p.add_argument("--variant", default="yunet_n", choices=["yunet_n", "yunet_s"])
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--epochs", type=int, default=560)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--warmup-iters", type=int, default=1500)
    p.add_argument("--warmup-lr", type=float, default=0.001)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--milestones", type=int, nargs="+", default=[400, 544],
                   help="StepLR 衰减 epoch（论文: 400, 544）")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--work-dir", default="work_dirs/yunet_n")
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--resume", default="")
    p.add_argument("--limit", type=int, default=0, help="仅用前 N 个样本（冒烟测试）")
    p.add_argument("--smoke", action="store_true", help="CPU 冒烟: 2 epoch/64 样本/64 尺寸")
    return p.parse_args()


def make_loader(args, split: str) -> DataLoader:
    img_dir = os.path.join(args.data_root, f"WIDER_{split}", "images")
    ann = os.path.join(args.data_root, "labelv2", split, "labelv2.txt")
    ds = WIDERFaceDataset(ann, img_dir, train=(split == "train"),
                          image_size=args.image_size)
    if args.limit and split == "train":
        ds.records = ds.records[: args.limit]
    return DataLoader(ds, batch_size=args.batch_size,
                      shuffle=(split == "train"), num_workers=args.workers,
                      collate_fn=collate, pin_memory=args.device.startswith("cuda"),
                      drop_last=(split == "train"))


def main():
    args = parse_args()
    if args.smoke:
        args.epochs, args.limit, args.image_size = 2, 64, 64
        args.batch_size, args.workers, args.device = 4, 0, "cpu"
        args.work_dir = "work_dirs/smoke"
    os.makedirs(args.work_dir, exist_ok=True)
    device = torch.device(args.device)

    model = build_yunet(args.variant).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {args.variant}: {n_params:,} params (paper yunet_n = 75,856)")

    criterion = YuNetCriterion()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)

    train_loader = make_loader(args, "train")
    start_epoch = 1
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[resume] from {args.resume} @ epoch {start_epoch}")

    global_step = (start_epoch - 1) * len(train_loader)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        # StepLR: 论文在第 400 / 544 epoch 衰减 x0.1
        decays = sum(1 for m in args.milestones if epoch > m)
        base_lr = args.lr * (0.1 ** decays)

        t0, running, seen = time.time(), 0.0, 0
        for it, batch in enumerate(train_loader):
            global_step += 1
            if global_step <= args.warmup_iters:
                lr = args.warmup_lr + (args.lr - args.warmup_lr) * global_step / args.warmup_iters
            else:
                lr = base_lr
            for g in optimizer.param_groups:
                g["lr"] = lr

            images = batch["images"].to(device, non_blocking=True)
            boxes = [b.to(device) for b in batch["boxes"]]
            labels = [l.to(device) for l in batch["labels"]]
            kps = [k.to(device) for k in batch["keypoints"]]

            losses = criterion(model(images), boxes=boxes, labels=labels, keypoints=kps)
            loss = sum(losses.values())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 35.0)
            optimizer.step()

            running += loss.item()
            seen += 1
            if it % args.log_interval == 0:
                parts = " ".join(f"{k}={v.item():.4f}" for k, v in losses.items())
                speed = seen / max(time.time() - t0, 1e-6)
                print(f"epoch={epoch} step={it + 1}/{len(train_loader)} "
                      f"loss={running / seen:.4f} lr={lr:.5f} {speed:.2f}it/s | {parts}",
                      flush=True)

        ckpt = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    epoch=epoch, variant=args.variant, args=vars(args))
        torch.save(ckpt, os.path.join(args.work_dir, "latest.pth"))
        print(f"[ckpt] epoch {epoch} saved -> latest.pth "
              f"(epoch time {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
