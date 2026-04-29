import torch
import torch.nn.functional as F
from mmcv.ops import point_sample
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou


def batch_dice_loss(inputs, targets):
    inputs = inputs.sigmoid().flatten(1)
    numerator = 2 * torch.einsum('nc,mc->nm', inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    return 1 - (numerator + 1) / (denominator + 1)


def batch_sigmoid_ce_loss(inputs, targets):
    hw = inputs.shape[1]
    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction='none')
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction='none')
    loss = torch.einsum('nc,mc->nm', pos, targets) + torch.einsum(
        'nc,mc->nm', neg, (1 - targets))
    return loss / hw


class HungarianMatcher(nn.Module):
    def __init__(self,
                 cost_class=4.0,
                 cost_mask=5.0,
                 cost_dice=5.0,
                 num_points=12544,
                 cost_box=5.0,
                 cost_giou=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.cost_box = cost_box
        self.cost_giou = cost_giou
        self.num_points = num_points
        assert cost_class or cost_mask or cost_dice or cost_box or cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets, cost=('cls', 'box', 'mask')):
        bs, num_queries = outputs['pred_logits'].shape[:2]
        indices = []
        for b in range(bs):
            out_bbox = outputs['pred_boxes'][b]
            tgt_bbox = targets[b]['boxes']
            if tgt_bbox.shape[0] == 0:
                indices.append((
                    torch.empty(0, dtype=torch.int64),
                    torch.empty(0, dtype=torch.int64),
                ))
                continue
            if 'box' in cost and tgt_bbox.numel() > 0:
                cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
                cost_giou = -generalized_box_iou(
                    box_cxcywh_to_xyxy(out_bbox),
                    box_cxcywh_to_xyxy(tgt_bbox))
            else:
                cost_bbox = out_bbox.new_zeros((num_queries, len(tgt_bbox)))
                cost_giou = out_bbox.new_zeros((num_queries, len(tgt_bbox)))

            out_prob = outputs['pred_logits'][b].sigmoid()
            tgt_ids = targets[b]['labels']
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = ((1 - alpha) * (out_prob ** gamma) *
                              (-(1 - out_prob + 1e-8).log()))
            pos_cost_class = (alpha * ((1 - out_prob) ** gamma) *
                              (-(out_prob + 1e-8).log()))
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            if 'mask' in cost and self.num_points > 0:
                out_mask = outputs['pred_masks'][b][:, None]
                tgt_mask = targets[b]['masks'].to(out_mask)[:, None]
                point_coords = torch.rand(1, self.num_points, 2,
                                          device=out_mask.device)
                tgt_mask = point_sample(
                    tgt_mask, point_coords.repeat(tgt_mask.shape[0], 1, 1),
                    align_corners=False).squeeze(1)
                out_mask = point_sample(
                    out_mask, point_coords.repeat(out_mask.shape[0], 1, 1),
                    align_corners=False).squeeze(1)
                with autocast(enabled=False):
                    out_mask = out_mask.float()
                    tgt_mask = tgt_mask.float()
                    cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask)
                    cost_dice = batch_dice_loss(out_mask, tgt_mask)
            else:
                cost_mask = out_bbox.new_zeros((num_queries, len(tgt_ids)))
                cost_dice = out_bbox.new_zeros((num_queries, len(tgt_ids)))

            c = (self.cost_mask * cost_mask + self.cost_class * cost_class +
                 self.cost_dice * cost_dice + self.cost_box * cost_bbox +
                 self.cost_giou * cost_giou)
            c = c.reshape(num_queries, -1).cpu()
            indices.append(linear_sum_assignment(c))

        return [(torch.as_tensor(i, dtype=torch.int64, device=outputs['pred_logits'].device),
                 torch.as_tensor(j, dtype=torch.int64, device=outputs['pred_logits'].device))
                for i, j in indices]
