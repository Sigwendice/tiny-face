# tiny-face

**极简端到端人脸识别系统：YuNet 检测 + MobileFaceNet 特征提取，两个模型合计 < 1.6M 参数，纯 CPU 完成训练与推理。**

覆盖 **数据 → 训练 → 推理 → 业务** 全链路，开箱即含可用权重与业务 API。

| | 检测器 | 识别器 |
|---|---|---|
| 模型 | YuNet (yunet_n) | MobileFaceNet + ArcFace |
| 参数量 | 75,856 | 1.50M |
| 输入 | 320×320 | 112×112 人脸 |
| 输出 | 人脸框 + 置信度 + 5 关键点 | 512 维 L2 归一化 embedding |
| 权重 | 官方预训练（WIDER FACE） | 本仓库 CPU 训练（LFW） |

**实测效果（纯 CPU）**

| 能力 | 指标 |
|---|---|
| 人脸检测 | 612 张图 0 失败 |
| 1:1 人脸验证 | acc **89.5%** / AUC **0.944**（600 对标定） |
| 1:N 以图搜图 | top-1 命中率 **92%**（4 人小库） |
| 特征提取速度 | CPU 单张 ~0.15s（检测+对齐+前向） |

## 目录

- [快速上手](#快速上手)
- [系统架构](#系统架构)
- [项目结构](#项目结构)
- [环境安装](#环境安装)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [单模型推理](#单模型推理)
- [业务使用](#业务使用face_servicepy)
- [实测效果与工程教训](#实测效果与工程教训)
- [FAQ](#faq)
- [已知限制与 Roadmap](#已知限制与-roadmap)
- [来源与致谢](#来源与致谢)

## 快速上手

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 1:1 人脸验证 —— 判断两张照片是否同一人
python face_service.py verify img_a.jpg img_b.jpg
# >> 相似度: 1.0000 (阈值 0.9788) -> 同一人

# 3. 建立人脸底库（扫描文件夹, 特征缓存）
python face_service.py build-gallery D:/photos -o gallery.npz

# 4. 以图搜图（1:N 检索）
python face_service.py search query.jpg gallery.npz -k 5
```

无需任何手动预处理：任意照片直接喂，YuNet 自动检测人脸并裁剪对齐。

## 系统架构

```
任意照片 ──▶ YuNet 人脸检测 ──▶ 检测框 + 2.8 倍边距裁剪 112×112 ──▶ MobileFaceNet
             (75,856 params)         (匹配训练分布, 见工程教训)          │
                                                                        ▼
                                                    512 维 L2 归一化特征 (ArcFace 训练)
                                                                        │
                                      ┌─────────────────────────────────┼──────────────────┐
                                      ▼                                 ▼                  ▼
                                  1:1 验证                        1:N 检索/聚类         特征入库
                              (余弦相似度 vs 阈值)              (内积排序 / npz 缓存)   (gallery.npz)
```

## 项目结构

```
tiny-face/
├── yunet/                  # YuNet 检测器包（model / engine / dataset，按论文复现）
├── facenet/                # 识别器包（MobileFaceNet + ArcFace + LFW 数据管道）
├── weights/                # YuNet 官方权重（pth + onnx），入库分发
├── checkpoints/            # 本仓库训练的 MobileFaceNet（best @ep147）
├── train_detector.py       # 训练 YuNet（WIDER FACE）
├── train_recognizer.py     # 训练 MobileFaceNet + ArcFace（LFW）
├── detect_faces.py         # 检测推理 demo（可视化 + ONNX 对照）
├── check_yunet.py          # YuNet 结构校验（参数量/前向/损失）
├── face_service.py         # ★ 业务入口：检测→对齐→特征→验证/检索
├── requirements.txt
└── README.md
```

## 环境安装

```bash
pip install -r requirements.txt   # torch / torchvision / opencv / numpy<2 / pillow
```

- 纯 CPU 即可运行全部功能；有 GPU 自动使用
- 已验证：Python 3.10 / torch 2.3.0+cpu / opencv 4.13 / numpy 1.26
- ⚠️ cv2 必须是完好的 opencv-python 构建：若 `import cv2` 成功但访问任何属性都
  报 `AttributeError`，说明安装损坏（空命名空间包），执行
  `pip install --force-reinstall opencv-python` 修复

## 数据准备

| 任务 | 数据集 | 组织形式 |
|---|---|---|
| 检测训练 | WIDER FACE | `data/widerface/WIDER_train/images/` + 官方 `labelv2/train/` 标注 |
| 识别训练 | LFW / CASIA-WebFace | `人名文件夹/图片.jpg`（一人一文件夹） |

```
data_dir/
├── person_A/
│   ├── person_A_0001.jpg
│   └── ...
└── person_B/
    └── ...
```

- LFW：13233 张 / 5749 身份，官方下载解压即是上述结构
- 换 **CASIA-WebFace**（51 万张 / 1 万身份）可显著提升识别质量——组织成同样
  结构，`--data_dir` 指过去即可，代码零修改
- 识别训练默认过滤图片数 < 10 的身份，原因见 [FAQ](#faq)

## 模型训练

### 训练识别器 MobileFaceNet + ArcFace

```bash
# 完整训练（本机实测 CPU ~1.7s/step, 300 epochs 约 16 小时）
python train_recognizer.py --data_dir D:/Download/lfw --epochs 300 --min_images 10

# 断点续训
python train_recognizer.py --epochs 300 --resume checkpoints/checkpoint_last.pth

# 冒烟测试（⚠️ 会覆盖 checkpoints/ 下的模型, 务必指定独立输出目录）
python train_recognizer.py --epochs 1 --limit_steps 5 --out_dir checkpoints_smoke
```

**训练配置**：ArcFace (s=64, m=0.5) + SGD (momentum 0.9, wd 5e-4)，lr 0.1 在
40%/70%/90% 进度处 ×0.1（里程碑按 `--epochs` 比例自动缩放）。

**评估协议**：按**身份**划分 train/val（同一人绝不跨集，杜绝泄漏），在从未参训
的身份上生成 600 对正负样本，报告最优阈值下的 verify_acc 与 AUC，best checkpoint
自动保存。

**本仓库训练结果**：300 epochs / CPU，best @ep147，val_acc 84.0% / AUC 0.896。

### 训练检测器 YuNet（可选，已提供官方权重）

```bash
python check_yunet.py                                       # 结构校验
python train_detector.py --data-root data/widerface --epochs 560
```

论文配置：simOTA 样本匹配，损失 = BCE(cls) + BCE(obj) + 5.0·EIoU(bbox) +
0.1·SmoothL1(kps)，lr 线性 warmup 1500 iters → 0.01，StepLR ×0.1 @400/544。
跳过训练直接使用 `weights/yunet_n.pth`（官方权重，已验证 strict=True 加载）。

### 训练量怎么定

人脸识别看**总迭代数**（充分训练约 5 万~15 万步）而非 epoch 数：

| 配置 | 总迭代 | 定位 |
|---|---|---|
| 本仓库（LFW 子集 + CPU） | ~3.4 万步 | 验证链路 / 小场景可用 |
| CASIA-WebFace + GPU + batch 512 | 9~12 万步 | 接近开源 SOTA 水平 |

## 单模型推理

```bash
# YuNet 检测 demo（画框/关键点，支持 ONNX 对照）
python detect_faces.py --weights weights/yunet_n.pth --image demo.jpg --size 320

# MobileFaceNet 前向自测
python -c "from facenet import MobileFaceNet; import torch; \
  m = MobileFaceNet(); print(m(torch.randn(1,3,112,112)).shape)"   # [1, 512]
```

## 业务使用（face_service.py）

`FaceService` 把两个模型串成完整业务，任意照片直接喂：

```python
from face_service import FaceService

svc = FaceService("checkpoints/checkpoint_best.pth")   # 默认: YuNet + box(2.8) + th=0.9788

# 1:1 验证
svc.verify("img_a.jpg", "img_b.jpg")
# -> {'ok': True, 'sim': 1.0, 'threshold': 0.9788, 'is_same': True}

# 特征提取（512 维 torch 向量, 检测不到人脸返回 None）
emb = svc.extract("any_photo.jpg")

# 1:N 检索
svc.build_gallery("D:/photos", "gallery.npz")   # 一次性建库, 特征缓存
svc.search("query.jpg", "gallery.npz", top_k=5)
# -> [{'label': 'person_A', 'path': '...', 'sim': 0.99}, ...]
```

CLI 等价：

```bash
python face_service.py verify 图A.jpg 图B.jpg
python face_service.py build-gallery D:/photos -o gallery.npz
python face_service.py search 查询图.jpg gallery.npz -k 5
```

**关键参数**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--detector` | `yunet` | `yunet` / `haar`（YuNet 加载失败自动降级） |
| `--align` | `box` | `box`=检测框+边距裁剪（**默认必选**，匹配训练分布）；`arc`=5 点标准对齐（对本模型不适用，见下） |
| `--margin` | 2.8 | box 裁剪边距系数（实测 2.4~3.2 均可） |
| `--threshold` | 0.9788 | 余弦相似度判决阈值（600 对标定，业务请重标定） |
| `--yunet-size` | 320 | 检测分辨率（0=原图） |

## 实测效果与工程教训

| 业务能力 | 结果 |
|---|---|
| YuNet 检测 | 612 张图 0 失败（LFW + 真实照片） |
| 1:1 验证 | **acc 89.5% / AUC 0.944**（600 对标定，优于训练时原始裁剪评估 84%/0.896） |
| 1:N 检索 top-1 | **11/12（92%）**（4 人小库） |

> **本项目最重要的工程教训：推理预处理必须匹配训练分布。**
>
> 本识别模型用 LFW 原始松裁剪（无对齐）训练。部署时直觉上应该做"标准 5 点
> ArcFace 相似变换对齐"——实测反而把 1:1 准确率从 84% 拖到 **63%**（分布偏移，
> 模型从未见过标准对齐人脸）。改用"检测框 + 2.8 倍边距方形裁剪"（贴近训练
> 分布）后，1:1 与 1:N 同时大幅提升。margin 扫描数据：
>
> | margin | 1.0 | 1.6 | 2.0 | 2.4 | **2.8** | 3.2 |
> |---|---|---|---|---|---|---|
> | 1:1 acc | 54.5% | 76.8% | 87.8% | 89.5% | **89.5%** | 90.0% |
>
> 如果你要用更大规模数据重训识别器（如 CASIA-WebFace + 标准对齐），训练管线
> 也应加入同样的对齐，届时 `--align arc` 才是正确选择。

## FAQ

**Q: 为什么默认过滤只有 1 张图的身份？**
分类损失（ArcFace）下，1 张图的类只学得到"记住这一张"，学不到类内变化；
且 LFW 70% 的身份是单图，全部纳入会把 ArcFace 头稀释成 96% 的单图类，噪声
远大于收益。改用 metric loss（pair/triplet）时单图身份可复用（当 anchor/
negative）。

**Q: 为什么阈值是 0.9788 这么高？**
该模型 embedding 整体分布收紧（同人 sim 均值 0.99），阈值自然落在高位。它是
LFW 600 对上标定的最优值，换业务场景请用自己的正负样本重算。

**Q: 能直接用到生产吗？**
链路与工程实践（防泄漏划分、阈值标定方法、分布匹配原则）可直接复用；但当前
识别权重受限于训练数据，仍有跨人误接受，生产请换 CASIA-WebFace/GlintMBA 级
数据重训后再上线。

## 已知限制与 Roadmap

1. **识别模型训练数据太小**：LFW 本质是评测集（5749 人中 70% 单图），当前模型
   偶有跨人误接受。→ Roadmap：CASIA-WebFace 重训
2. **迭代量不足**：3.4 万步 vs 标准 9~12 万步
3. **大角度侧脸/遮挡**：检测端 YuNet 鲁棒，但识别模型只见过 LFW 正脸分布
4. **阈值未做业务级标定**：提供标定方法（600 对采样 + 遍历阈值取最优 acc），
   部署前请用自己场景的数据执行
5. **对齐策略与训练绑定**：更换训练数据/对齐方式时，`face_service.py --align`
   需同步调整

## 来源与致谢

- **YuNet**: W. Wu, H. Peng, S. Yu. *YuNet: A Tiny Millisecond-level Face
  Detector*, Machine Intelligence Research 2023。官方权重来自
  [libfacedetection.train](https://github.com/ShiqiYu/libfacedetection.train)；
  本仓库 `yunet/` 为按论文的 PyTorch 复现（结构逐层对齐，官方权重 strict=True 加载）。
- **MobileFaceNet**: Chen et al., *MobileFaceNets: Efficient CNNs for Accurate
  Face Recognition*
- **ArcFace**: Deng et al., CVPR 2019（InsightFace 数值稳定实现）
- **数据集**: LFW (Huang et al. 2007) · WIDER FACE (Yang et al. 2016) · CASIA-WebFace (Yi et al. 2014)
