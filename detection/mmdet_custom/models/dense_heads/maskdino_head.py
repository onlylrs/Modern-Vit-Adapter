import numpy as np
import torch
import torch.nn.functional as F
from mmdet.core import INSTANCE_OFFSET, bbox2result
from mmdet.models.builder import HEADS
from mmdet.models.utils import preprocess_panoptic_gt
from torch import nn

from ..maskdino import HungarianMatcher, MaskDINOEncoder, MaskDINODecoder
from ..maskdino import SetCriterion
from ..maskdino import box_ops


@HEADS.register_module()
class MaskDINOHead(nn.Module):
    def __init__(self,
                 in_channels,
                 strides=(4, 8, 16, 32),
                 num_classes=80,
                 num_things_classes=None,
                 num_stuff_classes=0,
                 feat_channels=256,
                 mask_dim=256,
                 num_queries=300,
                 pixel_decoder=None,
                 transformer_decoder=None,
                 train_cfg=None,
                 test_cfg=None,
                 loss_cfg=None,
                 init_cfg=None):
        super().__init__()
        del init_cfg
        self.in_channels = in_channels
        self.strides = strides
        self.num_classes = num_classes
        self.num_things_classes = (
            num_classes if num_things_classes is None else num_things_classes)
        self.num_stuff_classes = num_stuff_classes
        self.num_queries = num_queries
        self.train_cfg = train_cfg or {}
        self.test_cfg = test_cfg or {}

        pixel_decoder = pixel_decoder or {}
        self.pixel_decoder = MaskDINOEncoder(
            in_channels=in_channels,
            strides=strides,
            conv_dim=feat_channels,
            mask_dim=mask_dim,
            **pixel_decoder,
        )
        transformer_decoder = transformer_decoder or {}
        self.predictor = MaskDINODecoder(
            in_channels=feat_channels,
            num_classes=num_classes,
            hidden_dim=feat_channels,
            num_queries=num_queries,
            mask_dim=mask_dim,
            **transformer_decoder,
        )

        loss_cfg = loss_cfg or {}
        losses = loss_cfg.get('losses', ['labels', 'masks', 'boxes'])
        matcher = HungarianMatcher(
            cost_class=loss_cfg.get('cost_class', 4.0),
            cost_mask=loss_cfg.get('cost_mask', 5.0),
            cost_dice=loss_cfg.get('cost_dice', 5.0),
            cost_box=loss_cfg.get('cost_box', 5.0),
            cost_giou=loss_cfg.get('cost_giou', 2.0),
            num_points=loss_cfg.get('num_points', 12544),
        )
        weight_dict = {
            'loss_cls': loss_cfg.get('class_weight', 4.0),
            'loss_mask': loss_cfg.get('mask_weight', 5.0),
            'loss_dice': loss_cfg.get('dice_weight', 5.0),
            'loss_bbox': loss_cfg.get('box_weight', 5.0),
            'loss_giou': loss_cfg.get('giou_weight', 2.0),
        }
        dec_layers = transformer_decoder.get('dec_layers', 9)
        dn = transformer_decoder.get('dn', 'no')
        if transformer_decoder.get('two_stage', True):
            weight_dict.update({f'interm.{k}': v for k, v in weight_dict.items()})
        if dn != 'no':
            dn_weight_dict = {
                f'{k}_dn': v
                for k, v in weight_dict.items()
                if not k.startswith('interm.')
            }
            weight_dict.update(dn_weight_dict)
        aux_weight_dict = {}
        for i in range(dec_layers):
            aux_weight_dict.update({f'd{i}.{k}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
        self.criterion = SetCriterion(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=loss_cfg.get('no_object_weight', 0.1),
            losses=losses,
            num_points=loss_cfg.get('num_points', 12544),
            oversample_ratio=loss_cfg.get('oversample_ratio', 3.0),
            importance_sample_ratio=loss_cfg.get('importance_sample_ratio', 0.75),
            dn=dn,
            dn_losses=loss_cfg.get('dn_losses', losses),
            semantic_ce_loss=loss_cfg.get('semantic_ce_loss', False),
        )

    def forward(self, feats, img_metas=None, targets=None):
        mask_features, _, multi_scale_features = self.pixel_decoder.forward_features(
            feats, masks=None)
        return self.predictor(multi_scale_features, mask_features, targets=targets)

    def _mask_to_tensor(self, masks, img_meta, device):
        h_pad, w_pad = img_meta['batch_input_shape']
        if masks is None:
            return torch.zeros((0, h_pad, w_pad), dtype=torch.float32,
                               device=device)
        mask_tensor = masks.to_tensor(dtype=torch.float32, device=device)
        padded = mask_tensor.new_zeros((mask_tensor.shape[0], h_pad, w_pad))
        h = min(mask_tensor.shape[-2], h_pad)
        w = min(mask_tensor.shape[-1], w_pad)
        padded[:, :h, :w] = mask_tensor[:, :h, :w]
        return padded

    def prepare_targets(self, gt_bboxes, gt_labels, gt_masks, img_metas, device):
        targets = []
        for i, (bboxes, labels) in enumerate(zip(gt_bboxes, gt_labels)):
            masks = None if gt_masks is None else gt_masks[i]
            padded_masks = self._mask_to_tensor(masks, img_metas[i], device)
            img_h, img_w = img_metas[i]['img_shape'][:2]
            scale = bboxes.new_tensor([img_w, img_h, img_w, img_h])
            boxes = box_ops.box_xyxy_to_cxcywh(bboxes) / scale
            boxes = boxes.clamp(min=0.0, max=1.0)
            targets.append({
                'labels': labels.to(device),
                'masks': padded_masks,
                'boxes': boxes.to(device),
            })
        return targets

    def prepare_panoptic_targets(self, gt_labels, gt_masks, gt_semantic_seg,
                                 img_metas, device):
        targets = []
        if isinstance(gt_semantic_seg, torch.Tensor):
            gt_semantic_seg = [seg for seg in gt_semantic_seg]
        for labels, masks, sem_seg, meta in zip(gt_labels, gt_masks,
                                                gt_semantic_seg, img_metas):
            labels, mask_tensor = preprocess_panoptic_gt(
                labels.to(device), masks, sem_seg.to(device),
                self.num_things_classes, self.num_stuff_classes)
            mask_tensor = mask_tensor.float()
            img_h, img_w = meta['img_shape'][:2]
            boxes_xyxy = box_ops.masks_to_boxes_bitmask(mask_tensor > 0)
            scale = boxes_xyxy.new_tensor([img_w, img_h, img_w, img_h])
            boxes = box_ops.box_xyxy_to_cxcywh(boxes_xyxy) / scale
            targets.append({
                'labels': labels.long(),
                'masks': mask_tensor,
                'boxes': boxes.clamp(min=0.0, max=1.0),
            })
        return targets

    def forward_train(self,
                      feats,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks=None,
                      gt_bboxes_ignore=None,
                      gt_semantic_seg=None):
        assert gt_bboxes_ignore is None
        device = feats[0].device
        if self.num_stuff_classes > 0 and gt_semantic_seg is not None:
            targets = self.prepare_panoptic_targets(
                gt_labels, gt_masks, gt_semantic_seg, img_metas, device)
        else:
            targets = self.prepare_targets(gt_bboxes, gt_labels, gt_masks,
                                           img_metas, device)
        outputs, mask_dict = self(feats, img_metas, targets)
        losses = self.criterion(outputs, targets, mask_dict)
        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] = losses[k] * self.criterion.weight_dict[k]
            else:
                losses.pop(k)
        return losses

    def simple_test(self, feats, img_metas, rescale=False):
        outputs, _ = self(feats, img_metas, targets=None)
        mask_cls_results = outputs['pred_logits']
        mask_pred_results = outputs['pred_masks']
        mask_box_results = outputs['pred_boxes']
        batch_h, batch_w = img_metas[0]['batch_input_shape']
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(batch_h, batch_w),
            mode='bilinear',
            align_corners=False,
        )

        results = []
        semantic_on = self.test_cfg.get('semantic_on', False)
        panoptic_on = self.test_cfg.get('panoptic_on', False)
        instance_on = self.test_cfg.get('instance_on', True)
        for mask_cls, mask_pred, mask_box, meta in zip(
                mask_cls_results, mask_pred_results, mask_box_results, img_metas):
            img_h, img_w = meta['img_shape'][:2]
            ori_h, ori_w = meta['ori_shape'][:2]
            mask_pred = mask_pred[:, :img_h, :img_w]
            boxes = self._box_postprocess(mask_box, img_h, img_w)
            if rescale:
                scale_factor = boxes.new_tensor(meta['scale_factor'])
                boxes = boxes / scale_factor
                mask_pred = F.interpolate(
                    mask_pred.unsqueeze(1), size=(ori_h, ori_w),
                    mode='bilinear', align_corners=False).squeeze(1)
            if panoptic_on:
                pan_results = self.panoptic_inference(mask_cls, mask_pred)
                results.append({'pan_results': pan_results})
                continue

            result = {}
            if semantic_on:
                result['sem_results'] = self.semantic_inference(
                    mask_cls, mask_pred).detach().cpu().numpy()
            if instance_on:
                det_bboxes, det_labels, mask_results = self.instance_inference(
                    mask_cls, mask_pred, boxes)
                bbox_results = bbox2result(det_bboxes, det_labels,
                                           self.num_classes)
                if semantic_on:
                    result['ins_results'] = (bbox_results, mask_results)
                else:
                    results.append((bbox_results, mask_results))
                    continue
            results.append(result)
        return results

    def semantic_inference(self, mask_cls, mask_pred):
        temperature = self.test_cfg.get('pano_temp', 0.06)
        transform_eval = self.test_cfg.get('transform_eval', True)
        mask_cls = mask_cls.sigmoid()
        if transform_eval:
            mask_cls = F.softmax(mask_cls / temperature, dim=-1)
        mask_pred = mask_pred.sigmoid()
        return torch.einsum('qc,qhw->chw', mask_cls, mask_pred)

    def panoptic_inference(self, mask_cls, mask_pred):
        object_mask_thr = self.test_cfg.get('object_mask_thr', 0.25)
        overlap_thr = self.test_cfg.get('overlap_threshold', 0.8)
        temperature = self.test_cfg.get('pano_temp', 0.06)
        transform_eval = self.test_cfg.get('transform_eval', True)

        scores, labels = mask_cls.sigmoid().max(-1)
        if transform_eval:
            scores, labels = F.softmax(
                mask_cls.sigmoid() / temperature, dim=-1).max(-1)
        keep = scores > object_mask_thr
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred.sigmoid()[keep]
        h, w = mask_pred.shape[-2:]
        panoptic_seg = torch.full((h, w), self.num_classes, dtype=torch.int32,
                                  device=mask_pred.device)
        if cur_masks.shape[0] == 0:
            return panoptic_seg.detach().cpu().numpy()

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks
        cur_mask_ids = cur_prob_masks.argmax(0)
        stuff_memory = {}
        instance_id = 1
        for k in range(cur_classes.shape[0]):
            pred_class = int(cur_classes[k].item())
            isthing = pred_class < self.num_things_classes
            mask_area = (cur_mask_ids == k).sum().item()
            original_area = (cur_masks[k] >= 0.5).sum().item()
            mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)
            if mask_area == 0 or original_area == 0 or mask.sum().item() == 0:
                continue
            if mask_area / original_area < overlap_thr:
                continue
            if not isthing:
                if pred_class in stuff_memory:
                    panoptic_seg[mask] = stuff_memory[pred_class]
                    continue
                stuff_memory[pred_class] = pred_class
                panoptic_seg[mask] = pred_class
            else:
                panoptic_seg[mask] = pred_class + instance_id * INSTANCE_OFFSET
                instance_id += 1
        return panoptic_seg.detach().cpu().numpy()

    def instance_inference(self, mask_cls, mask_pred, boxes):
        max_per_image = self.test_cfg.get('max_per_image', 100)
        focus_on_box = self.test_cfg.get('focus_on_box', False)
        scores = mask_cls.sigmoid()
        labels = torch.arange(self.num_classes, device=scores.device)
        labels = labels.unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(
            min(max_per_image, scores.numel()), sorted=False)
        labels_per_image = labels[topk_indices]
        query_indices = topk_indices // self.num_classes
        mask_pred = mask_pred[query_indices]
        boxes = boxes[query_indices]

        pred_masks = mask_pred > 0
        mask_scores = (
            mask_pred.sigmoid().flatten(1) * pred_masks.flatten(1)).sum(1) / (
                pred_masks.flatten(1).sum(1) + 1e-6)
        if focus_on_box:
            mask_scores = torch.ones_like(mask_scores)
        scores_per_image = scores_per_image * mask_scores
        det_bboxes = torch.cat([boxes, scores_per_image[:, None]], dim=1)

        mask_results = [[] for _ in range(self.num_classes)]
        pred_masks_np = pred_masks.detach().cpu().numpy().astype(np.uint8)
        labels_np = labels_per_image.detach().cpu().numpy()
        for mask, label in zip(pred_masks_np, labels_np):
            mask_results[int(label)].append(mask)
        return det_bboxes, labels_per_image, mask_results

    @staticmethod
    def _box_postprocess(out_bbox, img_h, img_w):
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        scale = torch.tensor([img_w, img_h, img_w, img_h],
                             dtype=boxes.dtype, device=boxes.device)
        boxes = boxes * scale
        return boxes
