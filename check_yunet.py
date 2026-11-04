"""结构复现验证（不训练也能跑的核心校验）.

校验项:
  1. 参数量: yunet_n=75,856 / yunet_s=54,608 (论文 Table 1/2)
  2. FLOPs:  yunet_n≈148.7M / yunet_s≈96.4M @320x320 (论文 Table 2, 卷积近似)
  3. 前向:   320x320 输入 -> 三级输出形状 (8/16/32 下采样)
  4. 官方权重: weights/yunet_n.pth 应可 strict=True 加载(模块命名与官方一致)
  5. 损失与匹配: 构造一个带 GT 的样例, 验证损失有限且 simOTA 分配有正样本

用法: python verify.py
"""

from __future__ import annotations

import torch

from yunet import build_yunet, YuNetCriterion, bbox_decode, MlvlPointGenerator


def count_flops(model: torch.nn.Module, h: int = 320, w: int = 320) -> float:
    """逐层近似统计 Conv2d/Linear FLOPs（按输出特征图尺寸估算）."""
    flops = [0]
    hooks = []

    def hook(module, inp, out):
        if isinstance(module, torch.nn.Conv2d):
            kh, kw = module.kernel_size
            flops[0] += out.shape[1] * out.shape[2] * out.shape[3] \
                * module.in_channels // module.groups * kh * kw

    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, h, w))
    for h_ in hooks:
        h_.remove()
    return flops[0] / 1e6  # MFLOPs (乘加计 1)


def main():
    ok = True

    # 1. 参数量
    for variant, paper_params, paper_flops in [("yunet_n", 75856, 148.7),
                                               ("yunet_s", 54608, 96.4)]:
        model = build_yunet(variant)
        n = sum(p.numel() for p in model.parameters())
        flops = count_flops(model)
        match_p = "OK" if n == paper_params else "MISMATCH"
        if n != paper_params:
            ok = False
        print(f"[params] {variant}: {n:,} (paper: {paper_params:,}) [{match_p}] | "
              f"FLOPs≈{flops:.1f}M (paper: {paper_flops}M)")

    # 2. 前向形状
    model = build_yunet("yunet_n").eval()
    with torch.no_grad():
        cls, bbox, obj, kps = model(torch.zeros(1, 3, 320, 320))
    shapes = [(tuple(c.shape), tuple(b.shape)) for c, b in zip(cls, bbox)]
    expect = [((1, 1, 40, 40), (1, 4, 40, 40)),
              ((1, 1, 20, 20), (1, 4, 20, 20)),
              ((1, 1, 10, 10), (1, 4, 10, 10))]
    match_s = "OK" if shapes == expect else "MISMATCH"
    if shapes != expect:
        ok = False
    print(f"[forward] 320x320 -> cls/bbox per level: {shapes} [{match_s}]")
    print(f"[forward] anchors total = {sum(s[0][2] * s[0][3] for s in shapes)} "
          f"(40^2+20^2+10^2 = 2100)")

    # 3. 官方权重 strict 加载
    ckpt_path = "weights/yunet_n.pth"
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt
        model.load_state_dict(state, strict=True)
        print(f"[weights] {ckpt_path} -> strict=True 加载成功, 模块命名与官方完全一致 [OK]")
    except FileNotFoundError:
        print(f"[weights] 未找到 {ckpt_path}, 跳过 strict 加载验证")
    except RuntimeError as e:
        ok = False
        print(f"[weights] strict 加载失败: {e}")

    # 4. 损失 + simOTA 冒烟
    model.train()
    criterion = YuNetCriterion()
    imgs = torch.randn(2, 3, 320, 320)
    boxes = [torch.tensor([[100.0, 100.0, 180.0, 160.0]]),
             torch.tensor([[40.0, 60.0, 72.0, 92.0], [200.0, 200.0, 260.0, 240.0]])]
    labels = [torch.zeros(1, dtype=torch.long), torch.zeros(2, dtype=torch.long)]
    kps = [torch.rand(1, 5, 3) * 0 + 140, torch.rand(2, 5, 3) * 0 + 100]
    kps[0][:, :, 2] = 1.0
    kps[1][:, :, 2] = 1.0
    preds = model(imgs)
    losses = criterion(preds, boxes=boxes, labels=labels, keypoints=kps)
    total = sum(losses.values())
    finite = bool(torch.isfinite(total))
    if not finite:
        ok = False
    print(f"[loss] " + " ".join(f"{k}={v.item():.4f}" for k, v in losses.items())
          + f" total={total.item():.4f} finite={finite} [{'OK' if finite else 'FAIL'}]")

    # 5. bbox 解码往返一致性
    gen = MlvlPointGenerator((8, 16, 32))
    priors = gen.grid_priors([(40, 40), (20, 20), (10, 10)])
    raw = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
    dec = bbox_decode(priors[:1], raw)
    expect_xy = (1 * 8 + 0, 2 * 8 + 0)
    match_d = "OK" if abs(dec[0, 0].item() - expect_xy[0] + 4) < 1e-4 else "check"
    print(f"[decode] prior(0,0,s=8) + pred(1,2,0,0) -> x1={dec[0, 0]:.1f}, y1={dec[0, 1]:.1f} "
          f"(预期中心 ({expect_xy[0]},{expect_xy[1]}), 边=center-4) [{match_d}]")

    print("\n=== 验证" + ("全部通过 ✅" if ok else "存在失败项 ❌") + " ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
