"""训练/推理引擎: 先验生成、编解码、EIoU 损失、simOTA 匹配、多任务损失、NMS、后处理.

对应论文:
  - Sec. 3.2.3  anchor-free 解码头 + simOTA 正样本匹配 + 多任务损失 (式 4)
  - EIoU loss:  Peng & Yu, "A Systematic IoU-Related Method", TIP 2021 (文献 [28])
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms as _tv_batched_nms


# ---------------------------------------------------------------------------
# 先验点生成 (stride 8/16/32 特征图上的格点, offset=0)
# ---------------------------------------------------------------------------
class MlvlPointGenerator:
    def __init__(self, strides: tuple[int, ...] = (8, 16, 32), offset: float = 0.0):
        self.strides = strides
        self.offset = offset

    def grid_priors(self, featmap_sizes: list[tuple[int, int]],
                    dtype=torch.float32, device="cpu") -> torch.Tensor:
        """返回 (N, 4): [cx, cy, stride_w, stride_h], 多级拼接."""
        all_points = []
        for lvl, (fh, fw) in enumerate(featmap_sizes):
            s = self.strides[lvl]
            sx = (torch.arange(fw, device=device) + self.offset) * s
            sy = (torch.arange(fh, device=device) + self.offset) * s
            yy, xx = torch.meshgrid(sy, sx, indexing="ij")
            pts = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
            stride = pts.new_full((pts.shape[0], 2), s)
            all_points.append(torch.cat([pts, stride], dim=-1))
        return torch.cat(all_points, dim=0).to(dtype)


# ---------------------------------------------------------------------------
# 编解码: bbox 预测 = (dx, dy, dw, dh), 相对格点、以 stride 为单位
# ---------------------------------------------------------------------------
def bbox_decode(priors: torch.Tensor, bbox_preds: torch.Tensor) -> torch.Tensor:
    """(N,4)[dx,dy,dw,dh] + priors(N,4)[cx,cy,sx,sy] -> (N,4)[x1,y1,x2,y2]."""
    centers = bbox_preds[..., :2] * priors[..., 2:] + priors[..., :2]
    whs = bbox_preds[..., 2:].exp() * priors[..., 2:]
    return torch.stack([
        centers[..., 0] - whs[..., 0] / 2,
        centers[..., 1] - whs[..., 1] / 2,
        centers[..., 0] + whs[..., 0] / 2,
        centers[..., 1] + whs[..., 1] / 2,
    ], dim=-1)


def kps_decode(priors: torch.Tensor, kps_preds: torch.Tensor) -> torch.Tensor:
    n = kps_preds.shape[-1] // 2
    return torch.cat([(kps_preds[..., [2 * i, 2 * i + 1]] * priors[..., 2:]) + priors[..., :2]
                      for i in range(n)], dim=-1)


def kps_encode(priors: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
    n = keypoints.shape[-1] // 2
    return torch.cat([(keypoints[..., [2 * i, 2 * i + 1]] - priors[..., :2]) / priors[..., 2:]
                      for i in range(n)], dim=-1)


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------
def bbox_overlaps(bboxes1: torch.Tensor, bboxes2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if bboxes1.numel() == 0 or bboxes2.numel() == 0:
        return bboxes1.new_zeros((bboxes1.size(0), bboxes2.size(0)))
    lt = torch.max(bboxes1[:, None, :2], bboxes2[None, :, :2])
    rb = torch.min(bboxes1[:, None, 2:], bboxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[..., 0] * wh[..., 1]
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    return overlap / (area1[:, None] + area2[None, :] - overlap + eps)


# ---------------------------------------------------------------------------
# EIoU loss（论文 Sec. 3.2.3, 文献[28]; smooth 变体, smooth_point=0.1）
# ---------------------------------------------------------------------------
def eiou_loss(pred: torch.Tensor, target: torch.Tensor, *,
              smooth_point: float = 0.1, eps: float = 1e-7,
              reduction: str = "none") -> torch.Tensor:
    px1, py1, px2, py2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    tx1, ty1, tx2, ty2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

    ex1, ey1 = torch.min(px1, tx1), torch.min(py1, ty1)
    ix1, iy1 = torch.max(px1, tx1), torch.max(py1, ty1)
    ix2, iy2 = torch.min(px2, tx2), torch.min(py2, ty2)
    xmin, ymin = torch.min(ix1, ix2), torch.min(iy1, iy2)
    xmax, ymax = torch.max(ix1, ix2), torch.max(iy1, iy2)

    inter = ((ix2 - ex1) * (iy2 - ey1) + (xmin - ex1) * (ymin - ey1)
             - (ix1 - ex1) * (ymax - ey1) - (xmax - ex1) * (iy1 - ey1))
    union = (px2 - px1) * (py2 - py1) + (tx2 - tx1) * (ty2 - ty1) - inter + eps
    ious = 1 - inter / union

    smooth_sign = (ious < smooth_point).detach().float()
    loss = (0.5 * smooth_sign * ious ** 2 / smooth_point
            + (1 - smooth_sign) * (ious - 0.5 * smooth_point))
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(reduction)


# ---------------------------------------------------------------------------
# simOTA 正样本匹配（论文 Sec. 3.2.3, 文献[26][27]）
# ---------------------------------------------------------------------------
class SimOTAAssigner:
    def __init__(self, *, center_radius: float = 2.5, candidate_topk: int = 10,
                 iou_weight: float = 3.0, cls_weight: float = 1.0):
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight

    @torch.no_grad()
    def assign(self, pred_scores, priors, decoded_bboxes, gt_bboxes, gt_labels):
        """pred_scores(N,C) 已过 sigmoid*objectness; priors(N,4); 返回 AssignResult."""
        inf = 100000.0
        num_gt = gt_bboxes.size(0)
        num_bboxes = decoded_bboxes.size(0)
        gt_inds = decoded_bboxes.new_full((num_bboxes,), 0, dtype=torch.long)

        in_center_or_box, is_in_both = self._in_gt_and_center_info(priors, gt_bboxes)
        valid_dec = decoded_bboxes[in_center_or_box]
        valid_scores = pred_scores[in_center_or_box]
        num_valid = valid_dec.size(0)

        max_overlaps = decoded_bboxes.new_zeros((num_bboxes,))
        if num_gt == 0 or num_bboxes == 0 or num_valid == 0:
            return gt_inds, max_overlaps

        pairwise_ious = bbox_overlaps(valid_dec, gt_bboxes)          # (Nv, G)
        iou_cost = -torch.log(pairwise_ious + 1e-7)

        gt_onehot = F.one_hot(gt_labels.to(torch.int64), pred_scores.shape[-1]).float()
        gt_onehot = gt_onehot.unsqueeze(0).repeat(num_valid, 1, 1)
        cls_cost = F.binary_cross_entropy(
            valid_scores.unsqueeze(1).repeat(1, num_gt, 1).float().sqrt(),
            gt_onehot, reduction="none").sum(-1)

        cost = cls_cost * self.cls_weight + iou_cost * self.iou_weight \
            + (~is_in_both) * inf
        matched_ious, matched_gt = self._dynamic_k(cost, pairwise_ious, num_gt, in_center_or_box)

        gt_inds[in_center_or_box] = matched_gt + 1
        max_overlaps[in_center_or_box] = matched_ious
        return gt_inds, max_overlaps

    def _in_gt_and_center_info(self, priors, gt_bboxes):
        num_gt = gt_bboxes.size(0)
        xs = priors[:, 0:1].repeat(1, num_gt)
        ys = priors[:, 1:2].repeat(1, num_gt)
        sw = priors[:, 2:3].repeat(1, num_gt)
        sh = priors[:, 3:4].repeat(1, num_gt)

        deltas = torch.stack([xs - gt_bboxes[:, 0], ys - gt_bboxes[:, 1],
                              gt_bboxes[:, 2] - xs, gt_bboxes[:, 3] - ys], dim=1)
        is_in_gts = deltas.min(dim=1).values > 0
        is_in_gts_any = is_in_gts.sum(dim=1) > 0

        cxs = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2
        cys = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2
        ct = torch.stack([xs - (cxs - self.center_radius * sw),
                          ys - (cys - self.center_radius * sh),
                          (cxs + self.center_radius * sw) - xs,
                          (cys + self.center_radius * sh) - ys], dim=1)
        is_in_cts = ct.min(dim=1).values > 0
        is_in_cts_any = is_in_cts.sum(dim=1) > 0

        in_any = is_in_gts_any | is_in_cts_any
        is_in_both = is_in_gts[in_any] & is_in_cts[in_any]
        return in_any, is_in_both

    def _dynamic_k(self, cost, pairwise_ious, num_gt, valid_mask):
        matching = torch.zeros_like(cost, dtype=torch.uint8)
        topk = min(self.candidate_topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, topk, dim=0)
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        for g in range(num_gt):
            _, pos_idx = torch.topk(cost[:, g], k=dynamic_ks[g], largest=False)
            matching[:, g][pos_idx] = 1

        multi_match = matching.sum(1) > 1
        if multi_match.sum() > 0:
            _, argmin = torch.min(cost[multi_match, :], dim=1)
            matching[multi_match, :] *= 0
            matching[multi_match, argmin] = 1

        fg_mask = matching.sum(1) > 0
        valid_mask[valid_mask.clone()] = fg_mask
        matched_gt = matching[fg_mask, :].argmax(1)
        matched_ious = (matching * pairwise_ious).sum(1)[fg_mask]
        return matched_ious, matched_gt


# ---------------------------------------------------------------------------
# 多任务损失（论文式 4: L = 1/N * sum(Lcls + 1.0*Lobj + 5.0*Lbbox + 0.1*Lldm)）
# ---------------------------------------------------------------------------
class YuNetCriterion:
    def __init__(self, *, num_classes: int = 1, strides=(8, 16, 32), kps_num: int = 5,
                 w_cls: float = 1.0, w_bbox: float = 5.0, w_obj: float = 1.0, w_kps: float = 0.1,
                 smooth_l1_beta: float = 1.0 / 9.0):
        self.num_classes = num_classes
        self.kps_num = kps_num
        self.prior_gen = MlvlPointGenerator(strides)
        self.assigner = SimOTAAssigner(center_radius=2.5)
        self.w = dict(cls=w_cls, bbox=w_bbox, obj=w_obj, kps=w_kps)
        self.smooth_l1_beta = smooth_l1_beta

    def __call__(self, preds, *, boxes, labels, keypoints):
        cls_scores, bbox_preds, obj_preds, kps_preds = preds
        num_imgs = cls_scores[0].shape[0]
        featmap_sizes = [tuple(c.shape[2:]) for c in cls_scores]
        priors = self.prior_gen.grid_priors(featmap_sizes, cls_scores[0].dtype,
                                            cls_scores[0].device)     # (N,4)
        flat_cls = _flatten(cls_scores, num_imgs, self.num_classes)
        flat_bbox = _flatten(bbox_preds, num_imgs, 4)
        flat_obj = _flatten(obj_preds, num_imgs, 1).squeeze(-1)
        flat_kps = _flatten(kps_preds, num_imgs, self.kps_num * 2)

        expanded = priors.unsqueeze(0).repeat(num_imgs, 1, 1)
        decoded = bbox_decode(expanded, flat_bbox)

        targets = [
            self._target_single(flat_cls[i].detach(), flat_obj[i].detach(),
                                expanded[i], decoded[i].detach(),
                                boxes[i], labels[i], keypoints[i])
            for i in range(num_imgs)]
        pos_mask = torch.cat([t["pos_mask"] for t in targets])
        cls_t = torch.cat([t["cls_target"] for t in targets])
        obj_t = torch.cat([t["obj_target"] for t in targets])
        bbox_t = torch.cat([t["bbox_target"] for t in targets])
        kps_t = torch.cat([t["kps_target"] for t in targets])
        kps_w = torch.cat([t["kps_weight"] for t in targets])
        num_pos = max(sum(t["num_pos"] for t in targets), 1)

        loss_obj = F.binary_cross_entropy_with_logits(
            flat_obj.reshape(-1, 1), obj_t, reduction="sum") / num_pos * self.w["obj"]

        if pos_mask.any():
            loss_cls = F.binary_cross_entropy_with_logits(
                flat_cls.reshape(-1, self.num_classes)[pos_mask], cls_t,
                reduction="sum") / num_pos * self.w["cls"]
            loss_bbox = eiou_loss(decoded.reshape(-1, 4)[pos_mask], bbox_t,
                                  reduction="sum") / num_pos * self.w["bbox"]
            enc_kps = kps_encode(expanded.reshape(-1, 4)[pos_mask], kps_t)
            loss_kps = _smooth_l1(flat_kps.reshape(-1, self.kps_num * 2)[pos_mask],
                                  enc_kps, weight=kps_w, beta=self.smooth_l1_beta) * self.w["kps"]
        else:
            zero = flat_cls.sum() * 0
            loss_cls = loss_bbox = loss_kps = zero

        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox,
                    loss_obj=loss_obj, loss_kps=loss_kps)

    @torch.no_grad()
    def _target_single(self, cls_pred, obj_pred, priors, decoded, gt_bboxes, gt_labels, gt_kps):
        num_priors = priors.size(0)
        zero = dict(pos_mask=cls_pred.new_zeros(num_priors).bool(),
                    cls_target=cls_pred.new_zeros((0, self.num_classes)),
                    obj_target=cls_pred.new_zeros((num_priors, 1)),
                    bbox_target=cls_pred.new_zeros((0, 4)),
                    kps_target=cls_pred.new_zeros((0, self.kps_num * 2)),
                    kps_weight=cls_pred.new_zeros((0, 1)),
                    num_pos=0)
        if gt_labels.numel() == 0:
            return zero

        offset_priors = torch.cat(
            [priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1)
        gt_inds, max_ious = self.assigner.assign(
            cls_pred.sigmoid() * obj_pred.sigmoid().unsqueeze(1),
            offset_priors, decoded, gt_bboxes, gt_labels)

        pos_mask = gt_inds > 0
        pos_inds = torch.nonzero(pos_mask).squeeze(1)
        if pos_inds.numel() == 0:
            return zero
        assigned = gt_inds[pos_inds] - 1
        num_pos = pos_inds.numel()

        cls_target = F.one_hot(gt_labels[assigned], self.num_classes
                               ).to(cls_pred.dtype) * max_ious[pos_inds].unsqueeze(-1)
        obj_target = torch.zeros_like(obj_pred).unsqueeze(-1)
        obj_target[pos_inds] = 1
        return dict(
            pos_mask=pos_mask,
            cls_target=cls_target,
            obj_target=obj_target,
            bbox_target=gt_bboxes[assigned],
            kps_target=gt_kps[assigned, :, :2].reshape(-1, self.kps_num * 2),
            kps_weight=gt_kps[assigned, :, 2].mean(dim=1, keepdim=True),
            num_pos=num_pos)


def _flatten(preds: list[torch.Tensor], num_imgs: int, ch: int) -> torch.Tensor:
    return torch.cat([p.permute(0, 2, 3, 1).reshape(num_imgs, -1, ch) for p in preds], dim=1)


def _smooth_l1(pred, target, *, weight, beta):
    if pred.numel() == 0:
        return pred.sum() * 0
    diff = (pred - target).abs()
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    avg = weight.sum()
    if avg <= 0:
        return pred.sum() * 0
    return (loss * weight).sum() / (avg + torch.finfo(torch.float32).eps)


# ---------------------------------------------------------------------------
# NMS + 后处理
# ---------------------------------------------------------------------------
def batched_nms(boxes: torch.Tensor, scores: torch.Tensor, idxs: torch.Tensor,
                iou_threshold: float) -> torch.Tensor:
    return _tv_batched_nms(boxes, scores, idxs, iou_threshold)


class YuNetPostprocessor:
    def __init__(self, *, strides=(8, 16, 32), score_threshold: float = 0.02,
                 nms_threshold: float = 0.45, top_k: int = 5000):
        self.prior_gen = MlvlPointGenerator(strides)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k

    @torch.no_grad()
    def __call__(self, preds):
        cls_scores, bbox_preds, obj_preds, kps_preds = preds
        num_imgs = cls_scores[0].shape[0]
        featmap_sizes = [tuple(c.shape[2:]) for c in cls_scores]
        priors = self.prior_gen.grid_priors(featmap_sizes, cls_scores[0].dtype,
                                            cls_scores[0].device)
        flat_cls = _flatten(cls_scores, num_imgs, cls_scores[0].shape[1]).sigmoid()
        flat_bbox = _flatten(bbox_preds, num_imgs, 4)
        flat_obj = _flatten(obj_preds, num_imgs, 1).squeeze(-1).sigmoid()
        flat_kps = _flatten(kps_preds, num_imgs, kps_preds[0].shape[1])

        expanded = priors.unsqueeze(0).repeat(num_imgs, 1, 1)
        boxes_all = bbox_decode(expanded, flat_bbox)
        kps_all = kps_decode(expanded, flat_kps)

        results = []
        for i in range(num_imgs):
            max_scores, labels = flat_cls[i].max(dim=1)
            scores = max_scores * flat_obj[i]
            # 先粗筛减少 NMS 计算量
            pre = scores > self.score_threshold * 0.1
            pre = pre.nonzero().squeeze(1)[: self.top_k]
            scores = scores[pre]
            keep = scores >= self.score_threshold
            cand = pre[keep]
            boxes = boxes_all[i][cand]
            scores = scores[keep]
            labels = labels[cand]
            kps = kps_all[i][cand]

            keep_idx = batched_nms(boxes, scores, labels, self.nms_threshold)
            results.append(dict(boxes=boxes[keep_idx], scores=scores[keep_idx],
                                labels=labels[keep_idx], keypoints=kps[keep_idx]))
        return results
