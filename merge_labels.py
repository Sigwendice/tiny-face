# -*- coding: utf-8 -*-
"""合并 SCRFD labelv2 (带关键点) 与官方 WIDER bbx_gt 的 ignore 标记.

SCRFD 官方 labelv2 把无效脸(invalid=1)内嵌成普通 19 值行(关键点全 -1),
与"确实没有关键点标注的脸"不可区分, 训练时会被误当正样本(约 1.5% 噪声).
本脚本用官方 bbx_gt 的 invalid 标记把这些框改写成 5 值 ignore 行
(x1 y1 x2 y2 1), 其余行原样保留 —— dataset.parse_labelv2 可正确解析.

输入文件均只读; 输出写到 --out, 不修改任何源文件.

用法:
  python merge_labels.py \
      --official wider_face_split/wider_face_train_bbx_gt.txt \
      --scrfd D:/Download/scrfd_label/train/labelv2.txt \
      --out data/widerface/labelv2/train/labelv2.txt
"""
import argparse


def load_ignores(official_path: str) -> set:
    """从官方 bbx_gt 提取 invalid=1 的框, 返回 {(path, x1, y1, x2, y2)} (整数圆整)."""
    ignores, cur = set(), None
    with open(official_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.isdigit():        # face count 行
                continue
            if " " not in line and "/" in line:   # 图片路径行
                cur = line
                continue
            vals = [float(v) for v in line.split()]
            x1, y1, w, h = vals[0], vals[1], vals[2], vals[3]
            if w <= 0 or h <= 0:
                continue
            invalid = int(vals[7]) if len(vals) > 7 else 0
            if invalid == 1:
                ignores.add((cur, round(x1), round(y1),
                             round(x1 + w), round(y1 + h)))
    return ignores


def merge(scrfd_path: str, ignores: set, out_path: str) -> None:
    n_face, n_kps, n_ign = 0, 0, 0
    cur = None
    with open(scrfd_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for raw in fin:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                cur = line[1:].split()[0]
                fout.write(line + "\n")
                continue
            v = [float(x) for x in line.split()]
            key = (cur, round(v[0]), round(v[1]), round(v[2]), round(v[3]))
            if key in ignores:
                fout.write(f"{v[0]:.0f} {v[1]:.0f} {v[2]:.0f} {v[3]:.0f} 1\n")
                n_ign += 1
            else:
                fout.write(line + "\n")
                n_face += 1
                if any(x != -1 for x in v[4:19]):
                    n_kps += 1
    print(f"[done] 有效脸 {n_face} (带关键点 {n_kps}), ignore 行 {n_ign}")
    print(f"[out] {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="合并 SCRFD 关键点标注与官方 ignore 标记")
    ap.add_argument("--official", required=True, help="官方 wider_face_*_bbx_gt.txt")
    ap.add_argument("--scrfd", required=True, help="SCRFD labelv2 标注 (只读)")
    ap.add_argument("--out", required=True, help="输出路径")
    a = ap.parse_args()
    ig = load_ignores(a.official)
    print(f"[official] ignore 框 {len(ig)} 个")
    merge(a.scrfd, ig, a.out)
