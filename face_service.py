# -*- coding: utf-8 -*-
"""人脸识别业务服务 —— 真实场景使用入口

完整业务链路: 「YuNet 人脸检测 -> ArcFace 5 点对齐 -> MobileFaceNet 特征提取
-> 验证/检索」。检测器默认用 YuNet（75,856 参数, 带置信度与 5 关键点,
权重来自 yunet-pytorch 官方权重），不可用时自动降级 Haar 级联。

用法:
  # 1:1 人脸验证
  python face_service.py verify img_a.jpg img_b.jpg

  # 构建 1:N 底库（扫描文件夹内所有图片, 缓存特征）
  python face_service.py build-gallery D:/photos -o gallery.npz

  # 1:N 检索（以图搜图, 返回 top-k）
  python face_service.py search query.jpg gallery.npz -k 5

  # 自定义判决阈值（默认 0.9788, 在 YuNet+box 对齐管线下 600 对标定 acc≈89.5%）
  python face_service.py verify img_a.jpg img_b.jpg --threshold 0.96
"""
import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from facenet import EVAL_TF
from facenet import MobileFaceNet

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# YuNet 项目根目录（相对本文件, 用于 import 模型与加载权重）
YUNET_ROOT = Path(__file__).resolve().parent

# ArcFace 112x112 标准 5 点模板: 左眼/右眼/鼻尖/左嘴角/右嘴角
ARC_TEMPLATE = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


class FaceService:
    """人脸识别业务服务: 检测对齐 + 特征提取 + 1:1 验证 + 1:N 检索"""

    def __init__(self, ckpt_path="checkpoints/checkpoint_best.pth",
                 threshold=0.9788, device=None,
                 detector="yunet", yunet_weights="", yunet_size=320,
                 align="box", margin=2.8):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model = MobileFaceNet(512)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()
        self.threshold = threshold

        # ---- 人脸检测器: YuNet 优先, 失败降级 Haar ----
        self.detector, self.yunet_size = detector, yunet_size
        self.align, self.margin = align, margin
        self.yunet = None
        if detector == "yunet":
            try:
                self.yunet = self._load_yunet(yunet_weights)
                print(f"[detector] YuNet ({self.yunet_size}x{self.yunet_size}, "
                      f"75,856 params)")
            except Exception as e:
                print(f"[detector] YuNet 加载失败({e}), 降级 Haar 级联")
        if self.yunet is None:
            self.detector = "haar"
        if self.detector == "haar":
            self._load_haar()
            print("[detector] Haar 级联")

    def _load_yunet(self, weights=""):
        sys.path.insert(0, str(YUNET_ROOT))
        from yunet import build_yunet
        from yunet.dataset import build_eval_transforms
        w = Path(weights) if weights else YUNET_ROOT / "weights" / "yunet_n.pth"
        if not w.exists():
            raise FileNotFoundError(str(w))
        model = build_yunet("yunet_n").to(self.device).eval()
        ckpt = torch.load(w, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt.get("model", ckpt))
        model.load_state_dict(state, strict=True)
        self.yunet_post = None  # 延迟导入, 见 _detect_yunet
        self._yunet_transforms = build_eval_transforms(self.yunet_size)
        return model

    def _load_haar(self):
        # Haar 级联检测器 (OpenCV 自带; 兼容 cv2.__file__ 为 None 的命名空间包构建)
        if hasattr(cv2, "data"):
            base = Path(cv2.data.haarcascades)
        else:
            import importlib.util
            spec = importlib.util.find_spec("cv2")
            base = Path(list(spec.submodule_search_locations)[0]) / "data"
        self.face_cascade = cv2.CascadeClassifier(str(base / "haarcascade_frontalface_default.xml"))
        self.eye_cascade = cv2.CascadeClassifier(str(base / "haarcascade_eye.xml"))
        if self.face_cascade.empty():
            raise RuntimeError(f"人脸检测器加载失败, 请检查 {base}")

    # ---------- 检测 ----------
    @torch.no_grad()
    def _detect_yunet(self, bgr, score_thr=0.6):
        """YuNet 检测, 返回 [{box:[x1,y1,x2,y2], score, kps:[[x,y]x5]}, ...]"""
        from yunet import YuNetPostprocessor
        h, w = bgr.shape[:2]
        sample = dict(image=bgr.copy())
        sample = self._yunet_transforms(sample)
        img = sample["image"].unsqueeze(0).to(self.device)
        scale = sample["scale"]
        preds = self.yunet(img)
        results = YuNetPostprocessor(score_threshold=score_thr,
                                     nms_threshold=0.45)(preds)[0]
        dets = []
        for box, score, kps in zip(results["boxes"], results["scores"],
                                   results["keypoints"]):
            x1, y1, x2, y2 = (box / scale).tolist()
            pts = (kps.reshape(5, 2) / scale).tolist()
            dets.append(dict(box=[x1, y1, x2, y2], score=float(score), kps=pts))
        return dets

    def detect_faces(self, bgr, score_thr=0.6):
        """统一检测入口: 返回 det 列表, 每项含 box/score/kps(可能为 None)"""
        if self.detector == "yunet":
            return self._detect_yunet(bgr, score_thr)
        # Haar 兜底: 只出框, 无关键点
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        return [dict(box=[x, y, x + w, y + h], score=1.0, kps=None)
                for x, y, w, h in faces]

    # ---------- 对齐 ----------
    def _align_by_kps(self, bgr, kps):
        """YuNet 5 关键点 -> ArcFace 112x112 相似变换对齐"""
        src = np.array(kps, dtype=np.float32)
        M, _ = cv2.estimateAffinePartial2D(src, ARC_TEMPLATE, method=cv2.LMEDS)
        if M is None:
            return None
        return cv2.warpAffine(bgr, M, (112, 112), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    def _detect_eyes(self, gray_roi):
        eyes = self.eye_cascade.detectMultiScale(gray_roi, 1.1, 5, minSize=(20, 20))
        if len(eyes) < 2:
            return None
        eyes = sorted(eyes, key=lambda e: e[0])
        left, right = eyes[0], eyes[-1]
        lc = (left[0] + left[2] / 2, left[1] + left[3] / 2)
        rc = (right[0] + right[2] / 2, right[1] + right[3] / 2)
        return lc, rc

    def _align_haar(self, bgr):
        """Haar 兜底: 最大脸 + 双眼水平对齐 -> 112x112"""
        faces = self.detect_faces(bgr)
        if not faces:
            return None
        x1, y1, x2, y2 = [int(v) for v in max(
            faces, key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))["box"]]
        x, y, w, h = x1, y1, x2 - x1, y2 - y1
        roi_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)[y:y + h, x:x + w]

        center, scale, angle = (x + w / 2, y + h / 2), 1.0, 0
        eye_data = self._detect_eyes(roi_gray)
        if eye_data is not None:
            (lx, ly), (rx, ry) = eye_data
            lx, ly, rx, ry = lx + x, ly + y, rx + x, ry + y
            angle = math.degrees(math.atan2(ry - ly, rx - lx))
            center = ((lx + rx) / 2, (ly + ry) / 2)
            eye_dist = max(math.hypot(rx - lx, ry - ly), 1e-6)
            scale = 112 * 0.85 / eye_dist
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += 112 / 2 - center[0]
        M[1, 2] += 112 / 2 - center[1]
        return cv2.warpAffine(bgr, M, (112, 112), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    def _crop_box(self, bgr, box, margin):
        """按检测框 + 边距系数做方形裁剪并 resize 112 (匹配训练分布)"""
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        s = max(x2 - x1, y2 - y1) * margin / 2
        H, W = bgr.shape[:2]
        x1c, y1c = int(max(0, cx - s)), int(max(0, cy - s))
        x2c, y2c = int(min(W, cx + s)), int(min(H, cy + s))
        if x2c - x1c < 10 or y2c - y1c < 10:
            return None
        return cv2.resize(bgr[y1c:y2c, x1c:x2c], (112, 112),
                          interpolation=cv2.INTER_CUBIC)

    def align_crop(self, bgr):
        """检测 + 对齐 -> 112x112 BGR, 失败返回 None

        align='box': YuNet 框 + margin 倍边距方形裁剪 (推荐, 匹配训练分布,
                     实测 600 对 acc≈89.5% / AUC≈0.944)
        align='arc': YuNet 5 关键点 -> ArcFace 标准模板相似变换 (分布偏移,
                     本模型实测仅 ~63%, 仅保留作选项)
        """
        if self.detector == "yunet":
            dets = self._detect_yunet(bgr)
            if not dets:
                return None
            best = max(dets, key=lambda d: d["score"])
            if self.align == "arc":
                return self._align_by_kps(bgr, best["kps"])
            return self._crop_box(bgr, best["box"], self.margin)
        return self._align_haar(bgr)

    # ---------- 特征提取 ----------
    @torch.no_grad()
    def extract(self, img):
        """img: 图片路径 / BGR ndarray / PIL Image -> 512 维 L2 归一化向量; 检测不到人脸返回 None"""
        if isinstance(img, (str, Path)):
            bgr = cv2.imdecode(np.fromfile(str(img), dtype=np.uint8),
                               cv2.IMREAD_COLOR)  # 兼容中文路径
            if bgr is None:
                raise FileNotFoundError(f"无法读取图片: {img}")
        elif isinstance(img, Image.Image):
            bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        else:
            bgr = img

        aligned = self.align_crop(bgr)
        if aligned is None:
            return None
        rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        x = EVAL_TF(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        return self.model(x).squeeze(0).cpu()

    # ---------- 1:1 验证 ----------
    def verify(self, img1, img2, threshold=None):
        """返回 dict(sim, is_same, ok)。任一图检测不到人脸时 ok=False"""
        th = self.threshold if threshold is None else threshold
        e1, e2 = self.extract(img1), self.extract(img2)
        if e1 is None or e2 is None:
            miss = [i for i, e in enumerate([e1, e2]) if e is None]
            return {"ok": False, "error": f"图 {','.join(str(i + 1) for i in miss)} 未检测到人脸"}
        sim = (e1 * e2).sum().item()
        return {"ok": True, "sim": round(sim, 4), "threshold": th,
                "is_same": bool(sim > th)}

    # ---------- 1:N 检索 ----------
    def build_gallery(self, folder, out_path="gallery.npz"):
        """扫描文件夹提取特征并缓存。按一级子文件夹名作为人物标签(若有)"""
        folder = Path(folder)
        items = []  # (label, path)
        for p in sorted(folder.rglob("*")):
            if p.suffix.lower() not in IMG_EXTS:
                continue
            label = p.parent.name if p.parent != folder else p.stem
            items.append((label, str(p)))
        if not items:
            raise FileNotFoundError(f"{folder} 下没有图片")

        embs, kept = [], []
        n_fail = 0
        for label, path in items:
            emb = self.extract(path)
            if emb is None:
                n_fail += 1
                continue
            embs.append(emb.numpy())
            kept.append((label, path))
        np.savez(out_path,
                 embs=np.stack(embs),
                 labels=np.array([k[0] for k in kept]),
                 paths=np.array([k[1] for k in kept]))
        print(f"[gallery] {len(kept)} 张入库成功, {n_fail} 张未检测到人脸 -> {out_path}")
        return out_path

    def search(self, query, gallery_path="gallery.npz", top_k=5):
        """以图搜图: 返回 [{label, path, sim}, ...] 按 sim 降序"""
        g = np.load(gallery_path, allow_pickle=False)
        q = self.extract(query)
        if q is None:
            raise RuntimeError("查询图未检测到人脸")
        sims = g["embs"] @ q.numpy()
        order = np.argsort(sims)[::-1][:top_k]
        return [{"label": str(g["labels"][i]), "path": str(g["paths"][i]),
                 "sim": round(float(sims[i]), 4)} for i in order]


def main():
    ap = argparse.ArgumentParser(description="人脸识别业务服务")
    ap.add_argument("--ckpt", default="checkpoints/checkpoint_best.pth")
    ap.add_argument("--threshold", type=float, default=0.9788)
    ap.add_argument("--detector", choices=["yunet", "haar"], default="yunet",
                    help="人脸检测器: yunet(默认) / haar(兜底)")
    ap.add_argument("--align", choices=["box", "arc"], default="box",
                    help="对齐方式: box=按检测框+边距裁剪(默认, 匹配训练分布) / "
                         "arc=5点ArcFace标准对齐(本模型不推荐)")
    ap.add_argument("--margin", type=float, default=2.8,
                    help="box 对齐的边距系数 (实测 2.4~3.2 均可)")
    ap.add_argument("--yunet-weights", default="",
                    help="YuNet 权重路径, 默认 ../yunet-pytorch/weights/yunet_n.pth")
    ap.add_argument("--yunet-size", type=int, default=320,
                    help="YuNet 推理分辨率 (320/640/origin)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_v = sub.add_parser("verify", help="1:1 人脸验证")
    p_v.add_argument("img1")
    p_v.add_argument("img2")

    p_b = sub.add_parser("build-gallery", help="构建 1:N 底库")
    p_b.add_argument("folder")
    p_b.add_argument("-o", "--out", default="gallery.npz")

    p_s = sub.add_parser("search", help="1:N 以图搜图")
    p_s.add_argument("query")
    p_s.add_argument("gallery")
    p_s.add_argument("-k", type=int, default=5)

    args = ap.parse_args()
    svc = FaceService(args.ckpt, args.threshold, detector=args.detector,
                      yunet_weights=args.yunet_weights,
                      yunet_size=None if args.yunet_size == 0 else args.yunet_size,
                      align=args.align, margin=args.margin)

    if args.cmd == "verify":
        r = svc.verify(args.img1, args.img2)
        if not r["ok"]:
            print(f"[失败] {r['error']}")
            return
        verdict = "同一人" if r["is_same"] else "不同人"
        print(f"相似度: {r['sim']:.4f} (阈值 {r['threshold']}) -> {verdict}")
    elif args.cmd == "build-gallery":
        svc.build_gallery(args.folder, args.out)
    elif args.cmd == "search":
        for rank, r in enumerate(svc.search(args.query, args.gallery, args.k), 1):
            print(f"top{rank}: sim={r['sim']:.4f} [{r['label']}] {r['path']}")


if __name__ == "__main__":
    main()
