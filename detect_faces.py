"""YuNet 推理 demo: 加载官方/自训权重, 对图片做人脸检测并可视化.

用法:
  python detect.py --weights weights/yunet_n.pth --image demo.jpg [--size 320|origin]
  python detect.py --onnx weights/yunet_n_320_320.onnx --image demo.jpg  # ONNX 对照

输出: 检测框 + 5 关键点绘制到 <image>_det.jpg, 终端打印人脸数与耗时。
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import torch

from yunet import build_yunet, YuNetPostprocessor
from yunet.dataset import build_eval_transforms


def load_model(weights: str, device: str) -> torch.nn.Module:
    model = build_yunet("yunet_n").to(device).eval()
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=True)
    print(f"[weights] loaded {weights} (epoch={ckpt.get('epoch', '?') if isinstance(ckpt, dict) else '?'})")
    return model


@torch.no_grad()
def detect(model: torch.nn.Module, image_bgr: np.ndarray, size: str,
           score_thr: float, nms_thr: float, device: str) -> tuple[list[dict], float]:
    h, w = image_bgr.shape[:2]
    sample = dict(image=image_bgr.copy())
    img_size = None if size == "origin" else int(size)
    sample = build_eval_transforms(img_size)(sample)
    img = sample["image"].unsqueeze(0).to(device)
    nh, nw = sample["img_shape"]
    scale, px, py = sample["scale"], 0, 0

    t0 = time.perf_counter()
    preds = model(img)
    results = YuNetPostprocessor(score_threshold=score_thr, nms_threshold=nms_thr)(preds)
    dt = (time.perf_counter() - t0) * 1000

    det = results[0]
    dets = []
    for box, score, kps in zip(det["boxes"], det["scores"], det["keypoints"]):
        x1, y1, x2, y2 = (box / scale).tolist()
        points = (kps.reshape(5, 2) / scale).tolist()
        dets.append(dict(box=[x1, y1, x2, y2], score=float(score), kps=points))
    return dets, dt


def draw(image: np.ndarray, dets: list[dict]) -> np.ndarray:
    vis = image.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d["box"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"{d['score']:.2f}", (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        for px, py in d["kps"]:
            cv2.circle(vis, (int(px), int(py)), 2, (0, 0, 255), -1)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/yunet_n.pth")
    ap.add_argument("--onnx", default="", help="可选: 用 ONNXRuntime 推理做对照")
    ap.add_argument("--image", required=True)
    ap.add_argument("--size", default="320", help="320 | 640 | origin")
    ap.add_argument("--score-thr", type=float, default=0.5)
    ap.add_argument("--nms-thr", type=float, default=0.45)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.image)

    if args.onnx:
        dets, dt = detect_onnx(args.onnx, image, args.size, args.score_thr, args.nms_thr)
    else:
        model = load_model(args.weights, args.device)
        dets, dt = detect(model, image, args.size, args.score_thr, args.nms_thr, args.device)

    print(f"[detect] {len(dets)} faces | {image.shape[1]}x{image.shape[0]} -> "
          f"size={args.size} | {dt:.1f} ms")
    for i, d in enumerate(dets[:10]):
        print(f"  face{i}: box=({d['box'][0]:.0f},{d['box'][1]:.0f},{d['box'][2]:.0f},"
              f"{d['box'][3]:.0f}) score={d['score']:.3f}")

    out = args.out or args.image.rsplit(".", 1)[0] + "_det.jpg"
    cv2.imwrite(out, draw(image, dets))
    print(f"[save] {out}")


def detect_onnx(onnx_path, image, size, score_thr, nms_thr):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    h, w = image.shape[:2]
    img_size = None if size == "origin" else int(size)
    if img_size:
        scale = min(img_size / w, img_size / h)
        nw, nh = int(w * scale + 0.5), int(h * scale + 0.5)
    else:
        scale, nw, nh = 1.0, w, h
    resized = cv2.resize(image, (nw, nh))
    padded = cv2.copyMakeBorder(resized, 0, (32 - nh % 32) % 32, 0, (32 - nw % 32) % 32,
                                cv2.BORDER_CONSTANT, value=0)
    blob = padded[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32)  # BGR->RGB?

    t0 = time.perf_counter()
    cls, obj, bbox, kps = sess.run(None, {inp_name: blob})
    dt = (time.perf_counter() - t0) * 1000
    # 官方 ONNX 输出顺序: cls, obj, bbox, kps; kps 已解码为绝对坐标(bbox 也是 x1y1x2y2?)
    # 注: 官方导出的 kps/bbox 是解码后的绝对坐标, 此处直接阈值+NMS
    cls = cls.reshape(-1)
    obj = obj.reshape(-1)
    bbox = bbox.reshape(-1, 4)
    kps = kps.reshape(-1, 10)
    scores = cls * obj
    keep = scores >= score_thr
    boxes, scores, kps = bbox[keep], scores[keep], kps[keep]
    import torchvision
    idx = torchvision.ops.nms(torch.from_numpy(boxes), torch.from_numpy(scores), nms_thr)
    dets = [dict(box=(b / scale).tolist(), score=float(s),
                 kps=(k.reshape(5, 2) / scale).tolist())
            for b, s, k in zip(boxes[idx.numpy()], scores[idx.numpy()], kps[idx.numpy()])]
    return dets, dt


if __name__ == "__main__":
    main()
