# -*- coding: utf-8 -*-
"""MobileFaceNet + ArcFace 人脸特征提取训练 (LFW)

用法:
  python train.py --data_dir D:/Download/lfw --epochs 30
  python train.py --data_dir D:/Download/lfw --min_images 5 --batch_size 16
  python train.py --resume checkpoint_last.pth
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from facenet import ArcFace
from facenet.lfw_dataset import (EVAL_TF, LFWDataset, dataset_stats, make_pairs,
                         scan_lfw, split_ids)
from facenet import MobileFaceNet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def extract_embeddings(model, paths, device, batch_size=64):
    """批量提取 embedding (评估用)"""
    from PIL import Image
    from facenet.lfw_dataset import EVAL_TF
    model.eval()
    embs = []
    for i in range(0, len(paths), batch_size):
        batch = [EVAL_TF(Image.open(p).convert("RGB"))
                 for p in paths[i:i + batch_size]]
        x = torch.stack(batch).to(device)
        embs.append(model(x).cpu())
    return torch.cat(embs)


@torch.no_grad()
def eval_verification(model, pairs, device):
    """验证集 pairs 上评估: 最优阈值准确率 + AUC"""
    p1 = extract_embeddings(model, [p[0] for p in pairs], device)
    p2 = extract_embeddings(model, [p[1] for p in pairs], device)
    sims = (p1 * p2).sum(dim=1).numpy()
    labels = np.array([p[2] for p in pairs])
    # 遍历候选阈值取最优准确率
    cands = np.unique(np.round(sims, 4))
    ths = np.concatenate([[cands[0] - 0.01], (cands[:-1] + cands[1:]) / 2,
                          [cands[-1] + 0.01]])
    accs = [((sims > t).astype(int) == labels).mean() for t in ths]
    best_acc = max(accs)
    best_th = ths[int(np.argmax(accs))]
    # AUC (rank-based)
    pos, neg = sims[labels == 1], sims[labels == 0]
    auc = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
    return best_acc, best_th, float(auc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="D:/Download/lfw")
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--min_images", type=int, default=10,
                    help="每个身份至少多少张图才纳入训练")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--embedding_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--val_every", type=int, default=2, help="每 N 个 epoch 验证一次")
    ap.add_argument("--log_every", type=int, default=20, help="每 N 个 step 打日志")
    ap.add_argument("--limit_steps", type=int, default=0,
                    help=">0 时每 epoch 只跑前 N 个 step (冒烟测试用)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", default="")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 数据 ----------
    by_id = scan_lfw(args.data_dir, args.min_images)
    print(f"[data] min_images={args.min_images}: {dataset_stats(by_id)}")
    train_ids, val_ids = split_ids(by_id, seed=args.seed)
    print(f"[data] train {dataset_stats(train_ids)} | val {dataset_stats(val_ids)}")

    train_set = LFWDataset(train_ids)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=False,
                              drop_last=True, persistent_workers=args.num_workers > 0)
    val_pairs = make_pairs(val_ids, n_pos=300, n_neg=300, seed=args.seed)
    print(f"[data] 训练样本 {len(train_set)}, 验证 pairs {len(val_pairs)}")

    # ---------- 模型 ----------
    num_classes = len(train_ids)
    model = MobileFaceNet(args.embedding_size).to(device)
    head = ArcFace(args.embedding_size, num_classes).to(device)
    print(f"[model] MobileFaceNet {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, "
          f"ArcFace classes={num_classes}, device={device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        [{"params": model.parameters()}, {"params": head.parameters()}],
        lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=sorted({max(1, int(args.epochs * r))
                                      for r in (0.4, 0.7, 0.9)}), gamma=0.1)

    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        head.load_state_dict(ckpt["head"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[resume] 从 epoch {start_epoch} 继续")

    best_acc = 0.0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        head.train()
        t0 = time.time()
        running_loss, running_correct, running_n = 0.0, 0, 0
        for step, (x, y) in enumerate(train_loader):
            if args.limit_steps and step >= args.limit_steps:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            emb = model(x)
            logits = head(emb, y)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.size(0)
            running_correct += (logits.argmax(1) == y).sum().item()
            running_n += y.size(0)
            if (step + 1) % args.log_every == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"[ep {epoch}] step {step + 1}/{len(train_loader)} "
                      f"loss={running_loss / running_n:.4f} "
                      f"acc={running_correct / running_n:.3f} lr={lr_now:g} "
                      f"({time.time() - t0:.0f}s)")
        scheduler.step()
        print(f"[ep {epoch}] epoch done: loss={running_loss / max(running_n, 1):.4f} "
              f"train_acc={running_correct / max(running_n, 1):.3f} "
              f"time={time.time() - t0:.0f}s")

        # ---------- 验证 + 存档 ----------
        is_last = epoch == args.epochs - 1
        if (epoch + 1) % args.val_every == 0 or is_last:
            acc, th, auc = eval_verification(model, val_pairs, device)
            print(f"[eval ep {epoch}] verify_acc={acc:.4f} (th={th:.3f}) AUC={auc:.4f}")
            if acc > best_acc:
                best_acc = acc
                torch.save({"model": model.state_dict(),
                            "head": head.state_dict(),
                            "epoch": epoch,
                            "val_acc": acc, "val_auc": auc,
                            "args": vars(args)},
                           out_dir / "checkpoint_best.pth")
                print(f"[save] 最佳模型 acc={acc:.4f} -> {out_dir / 'checkpoint_best.pth'}")

        torch.save({"model": model.state_dict(), "head": head.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch, "args": vars(args)},
                   out_dir / "checkpoint_last.pth")

    print(f"[done] 最佳验证准确率: {best_acc:.4f}")


if __name__ == "__main__":
    main()
