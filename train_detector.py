"""YuNet 训练脚本（论文 Sec. 4.2 训练配置）.

论文配置:
  - SGD, momentum 0.9, weight decay 5e-4
  - batch 16 x 2 GPU, lr 0.001 线性 warmup 1500 iters 至 0.01
  - StepLR: 第 400 / 544 epoch 衰减 x0.1, 共 560 epoch, 无预训练
  - 数据: WIDER FACE train (labelv2), 尺度增强 crop_choice=(0.5..1.5), 输入 640x640

用法:
  python train.py --data-root data/widerface --device cuda --batch-size 16
  python train.py --data-root data/widerface --epochs 2 --smoke   # 冒烟

每个 epoch 结束后在 WIDER_val 上做评测 (val loss + P/R/F1@IoU0.5),
F1 提升时保存 checkpoint_best.pth; 无 val 数据时自动跳过.
"""

from __future__ import annotations

import argparse
import math
import os
import time

import torch
from torchvision.ops import box_iou
from torch.utils.data import DataLoader

from yunet import build_yunet, YuNetCriterion, YuNetPostprocessor
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
    p.add_argument("--val-every", type=int, default=1,
                   help="每 N 个 epoch 评测一次验证集 (0 = 关闭)")
    p.add_argument("--val-limit", type=int, default=0, help="仅用前 N 张验证图 (0 = 全量)")
    p.add_argument("--score-thr", type=float, default=0.35,
                   help="验证评测的置信度阈值")
    p.add_argument("--smoke", action="store_true", help="CPU 冒烟: 2 epoch/64 样本/64 尺寸")
    return p.parse_args()


def make_loader(args, split: str) -> DataLoader:
    img_dir = os.path.join(args.data_root, f"WIDER_{split}", "images")
    ann = os.path.join(args.data_root, "labelv2", split, "labelv2.txt")
    ds = WIDERFaceDataset(ann, img_dir, train=(split == "train"),
                          image_size=args.image_size)
    limit = args.limit if split == "train" else args.val_limit
    if limit:
        ds.records = ds.records[:limit]
    return DataLoader(ds, batch_size=args.batch_size,
                      shuffle=(split == "train"), num_workers=args.workers,
                      collate_fn=collate, pin_memory=args.device.startswith("cuda"),
                      drop_last=(split == "train"))


def _match_dets(det_boxes, det_scores, gt_boxes, ign_boxes,
                score_thr: float, iou_thr: float):
    """贪心匹配一次检测. 返回 (tp, fp): 命中 ignore 框的检测不计入任何一方."""
    keep = det_scores >= score_thr
    det_boxes, det_scores = det_boxes[keep], det_scores[keep]
    order = det_scores.argsort(descending=True)
    det_boxes, det_scores = det_boxes[order], det_scores[order]

    matched_gt = torch.zeros(len(gt_boxes), dtype=torch.bool, device=det_boxes.device)
    tp = fp = discarded = 0
    for i in range(len(det_boxes)):
        box = det_boxes[i:i + 1]
        if len(ign_boxes):
            if float(box_iou(box, ign_boxes).max()) >= iou_thr:
                discarded += 1
                continue
        if len(gt_boxes):
            ious = box_iou(box, gt_boxes)[0]
            j = int(ious.argmax())
            if float(ious[j]) >= iou_thr and not matched_gt[j]:
                matched_gt[j] = True
                tp += 1
                continue
        fp += 1
    return tp, fp, discarded


@torch.no_grad()
def evaluate(model, val_loader, criterion, device, post,
             score_thr: float = 0.35, iou_thr: float = 0.5) -> dict:
    model.eval()
    tot_loss, n_batches = 0.0, 0
    tp = fp = fn = discarded = 0
    for batch in val_loader:
        images = batch["images"].to(device)
        # eval 路径 collate 出来是 numpy, 统一转张量
        boxes = [torch.as_tensor(b, device=device) for b in batch["boxes"]]
        labels = [torch.as_tensor(l, device=device) for l in batch["labels"]]
        kps = [torch.as_tensor(k, device=device) for k in batch["keypoints"]]

        preds = model(images)
        losses = criterion(preds, boxes=boxes, labels=labels, keypoints=kps)
        tot_loss += sum(losses.values()).item()
        n_batches += 1

        results = post(preds)
        for i, res in enumerate(results):
            gt = torch.as_tensor(batch["boxes"][i], device=device)
            ign = torch.as_tensor(batch["ignored"][i], device=device)
            d = _match_dets(res["boxes"], res["scores"], gt, ign,
                            score_thr, iou_thr)
            tp += d[0]; fp += d[1]; discarded += d[2]
            fn += len(gt)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    return dict(loss=tot_loss / max(n_batches, 1), precision=precision,
                recall=recall, f1=f1, tp=tp, fp=fp, fn=fn, ignored_hit=discarded)


def main():
    args = parse_args()
    if args.smoke:
        args.epochs, args.limit, args.image_size = 2, 64, 64
        args.batch_size, args.workers, args.device = 4, 0, "cpu"
        args.work_dir = "work_dirs/smoke"
        args.val_limit = args.val_limit or 16
    os.makedirs(args.work_dir, exist_ok=True)
    device = torch.device(args.device)

    model = build_yunet(args.variant).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {args.variant}: {n_params:,} params (paper yunet_n = 75,856)")

    criterion = YuNetCriterion()
    post = YuNetPostprocessor()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)

    train_loader = make_loader(args, "train")
    # val 数据缺失时自动跳过评测
    val_ann = os.path.join(args.data_root, "labelv2", "val", "labelv2.txt")
    val_img = os.path.join(args.data_root, "WIDER_val", "images")
    val_loader = None
    if args.val_every > 0 and os.path.isfile(val_ann) and os.path.isdir(val_img):
        val_loader = make_loader(args, "val")
        print(f"[val] {len(val_loader.dataset)} 张图, 每 {args.val_every} epoch 评测一次 "
              f"(score_thr={args.score_thr})")

    start_epoch = 1
    best_f1 = -1.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt.get("best_f1", -1.0)
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
                    epoch=epoch, variant=args.variant, args=vars(args),
                    best_f1=best_f1)
        torch.save(ckpt, os.path.join(args.work_dir, "latest.pth"))
        print(f"[ckpt] epoch {epoch} saved -> latest.pth "
              f"(epoch time {time.time() - t0:.1f}s)")

        if val_loader is not None and epoch % args.val_every == 0:
            m = evaluate(model, val_loader, criterion, device, post,
                         score_thr=args.score_thr)
            print(f"[eval] epoch={epoch} val_loss={m['loss']:.4f} "
                  f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} "
                  f"(tp={m['tp']} fp={m['fp']} fn={m['fn']} ign_hit={m['ignored_hit']})",
                  flush=True)
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                ckpt["best_f1"] = best_f1
                ckpt["val_metrics"] = m
                torch.save(ckpt, os.path.join(args.work_dir, "checkpoint_best.pth"))
                print(f"[ckpt] best F1={best_f1:.4f} -> checkpoint_best.pth")


if __name__ == "__main__":
    main()
