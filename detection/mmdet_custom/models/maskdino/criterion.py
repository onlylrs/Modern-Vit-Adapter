import torch
import torch.nn.functional as F
from mmcv.ops import point_sample
from mmdet.core import reduce_mean
from torch import nn

from . import box_ops
from .point_sample import get_uncertain_point_coords_with_randomness


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha=0.25, gamma=2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets,
                                                 reduction='none')
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / num_boxes


def dice_loss(inputs, targets, num_masks):
    inputs = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def sigmoid_ce_loss(inputs, targets, num_masks):
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    return loss.mean(1).sum() / num_masks


class SetCriterion(nn.Module):
    def __init__(self,
                 num_classes,
                 matcher,
                 weight_dict,
                 eos_coef,
                 losses,
                 num_points,
                 oversample_ratio,
                 importance_sample_ratio,
                 dn='no',
                 dn_losses=None,
                 semantic_ce_loss=False):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.focal_alpha = 0.25
        self.dn = dn
        self.dn_losses = [] if dn_losses is None else dn_losses
        self.semantic_ce_loss = semantic_ce_loss
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels_ce(self, outputs, targets, indices, num_boxes):
        src_logits = outputs['pred_logits'].float()
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t['labels'][j] for t, (_, j) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64,
            device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {'loss_cls': loss_ce}

    def loss_labels(self, outputs, targets, indices, num_boxes):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t['labels'][j] for t, (_, j) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64,
            device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(
            src_logits, target_classes_onehot, num_boxes,
            alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        return {'loss_cls': loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat(
            [t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        if target_boxes.numel() == 0:
            zero = src_boxes.sum() * 0
            return {'loss_bbox': zero, 'loss_giou': zero}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        return {
            'loss_bbox': loss_bbox.sum() / num_boxes,
            'loss_giou': loss_giou.sum() / num_boxes,
        }

    def loss_masks(self, outputs, targets, indices, num_masks):
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs['pred_masks'][src_idx]
        if sum(t['masks'].shape[0] for t in targets) == 0:
            zero = outputs['pred_masks'].sum() * 0
            return {'loss_mask': zero, 'loss_dice': zero}
        target_masks = torch.cat([
            t['masks'][j] for t, (_, j) in zip(targets, indices)
        ], dim=0).to(src_masks)

        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]
        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks, self.num_points, self.oversample_ratio,
                self.importance_sample_ratio)
            point_labels = point_sample(
                target_masks, point_coords, align_corners=False).squeeze(1)
        point_logits = point_sample(
            src_masks, point_coords, align_corners=False).squeeze(1)
        return {
            'loss_mask': sigmoid_ce_loss(point_logits, point_labels, num_masks),
            'loss_dice': dice_loss(point_logits, point_labels, num_masks),
        }

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(indices)
        ])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([
            torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)
        ])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels_ce
            if self.semantic_ce_loss else self.loss_labels,
            'masks': self.loss_masks,
            'boxes': self.loss_boxes,
        }
        return loss_map[loss](outputs, targets, indices, num_masks)

    def prep_for_dn(self, mask_dict):
        output_known_lbs_bboxes = mask_dict['output_known_lbs_bboxes']
        scalar = mask_dict['scalar']
        pad_size = mask_dict['pad_size']
        assert pad_size % scalar == 0
        single_pad = pad_size // scalar
        return output_known_lbs_bboxes, single_pad, scalar

    def forward(self, outputs, targets, mask_dict=None):
        outputs_without_aux = {k: v for k, v in outputs.items()
                               if k != 'aux_outputs'}
        exc_idx = None
        if self.dn != 'no' and mask_dict is not None:
            output_known_lbs_bboxes, single_pad, scalar = self.prep_for_dn(
                mask_dict)
            exc_idx = []
            device = next(iter(outputs.values())).device
            for target in targets:
                num_labels = len(target['labels'])
                if num_labels > 0:
                    tgt_idx = torch.arange(num_labels, device=device).long()
                    tgt_idx = tgt_idx.unsqueeze(0).repeat(scalar, 1).flatten()
                    output_idx = (
                        torch.arange(scalar, device=device).long().unsqueeze(1) *
                        single_pad +
                        torch.arange(num_labels, device=device).long()).flatten()
                else:
                    output_idx = torch.empty(0, dtype=torch.long, device=device)
                    tgt_idx = torch.empty(0, dtype=torch.long, device=device)
                exc_idx.append((output_idx, tgt_idx))
        indices = self.matcher(outputs_without_aux, targets)
        num_masks = sum(len(t['labels']) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float,
            device=next(iter(outputs.values())).device)
        num_masks = torch.clamp(reduce_mean(num_masks), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices,
                                        num_masks))

        if self.dn != 'no' and mask_dict is not None:
            dn_loss_dict = {}
            for loss in self.dn_losses:
                dn_loss_dict.update(
                    self.get_loss(loss, output_known_lbs_bboxes, targets,
                                  exc_idx, num_masks * scalar))
            dn_loss_dict = {f'{k}_dn': v for k, v in dn_loss_dict.items()}
            losses.update(dn_loss_dict)

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices,
                                           num_masks)
                    l_dict = {f'd{i}.{k}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                if self.dn != 'no' and mask_dict is not None:
                    for loss in self.dn_losses:
                        l_dict = self.get_loss(
                            loss, output_known_lbs_bboxes['aux_outputs'][i],
                            targets, exc_idx, num_masks * scalar)
                        l_dict = {
                            f'd{i}.{k}_dn': v
                            for k, v in l_dict.items()
                        }
                        losses.update(l_dict)

        if 'interm_outputs' in outputs:
            indices = self.matcher(outputs['interm_outputs'], targets)
            for loss in self.losses:
                l_dict = self.get_loss(loss, outputs['interm_outputs'], targets,
                                       indices, num_masks)
                l_dict = {f'interm.{k}': v for k, v in l_dict.items()}
                losses.update(l_dict)
        return losses
