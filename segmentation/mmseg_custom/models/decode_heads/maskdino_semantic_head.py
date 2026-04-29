import torch
import torch.nn.functional as F
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

from mmdet_custom.models.maskdino import (HungarianMatcher, MaskDINOEncoder,
                                          MaskDINODecoder, SetCriterion)
from mmdet_custom.models.maskdino import box_ops


@HEADS.register_module()
class MaskDINOSemanticHead(BaseDecodeHead):
    def __init__(self,
                 in_channels,
                 strides=(4, 8, 16, 32),
                 feat_channels=256,
                 mask_dim=256,
                 num_queries=100,
                 pixel_decoder=None,
                 transformer_decoder=None,
                 loss_cfg=None,
                 ignore_index=255,
                 init_cfg=None,
                 **kwargs):
        super().__init__(
            in_channels=in_channels,
            channels=feat_channels,
            num_classes=kwargs.pop('num_classes'),
            input_transform='multiple_select',
            ignore_index=ignore_index,
            init_cfg=init_cfg,
            **kwargs)
        self.strides = strides
        self.num_queries = num_queries
        self.ignore_index = ignore_index
        self.conv_seg = None

        pixel_decoder = pixel_decoder or {}
        self.pixel_decoder = MaskDINOEncoder(
            in_channels=in_channels,
            strides=strides,
            conv_dim=feat_channels,
            mask_dim=mask_dim,
            **pixel_decoder)

        transformer_decoder = transformer_decoder or {}
        transformer_decoder.setdefault('two_stage', False)
        transformer_decoder.setdefault('learn_tgt', False)
        transformer_decoder.setdefault('initialize_box_type', 'no')
        transformer_decoder.setdefault('semantic_ce_loss', True)
        self.predictor = MaskDINODecoder(
            in_channels=feat_channels,
            num_classes=self.num_classes,
            hidden_dim=feat_channels,
            num_queries=num_queries,
            mask_dim=mask_dim,
            **transformer_decoder)

        loss_cfg = loss_cfg or {}
        losses = loss_cfg.get('losses', ['labels', 'masks'])
        matcher = HungarianMatcher(
            cost_class=loss_cfg.get('cost_class', 4.0),
            cost_mask=loss_cfg.get('cost_mask', 5.0),
            cost_dice=loss_cfg.get('cost_dice', 5.0),
            cost_box=0.0,
            cost_giou=0.0,
            num_points=loss_cfg.get('num_points', 12544))
        weight_dict = {
            'loss_cls': loss_cfg.get('class_weight', 4.0),
            'loss_mask': loss_cfg.get('mask_weight', 5.0),
            'loss_dice': loss_cfg.get('dice_weight', 5.0),
        }
        dec_layers = transformer_decoder.get('dec_layers', 9)
        dn = transformer_decoder.get('dn', 'no')
        if dn != 'no':
            weight_dict.update({
                f'{k}_dn': v
                for k, v in weight_dict.items()
            })
        aux_weight_dict = {}
        for i in range(dec_layers):
            aux_weight_dict.update({
                f'd{i}.{k}': v
                for k, v in weight_dict.items()
            })
        weight_dict.update(aux_weight_dict)
        self.criterion = SetCriterion(
            num_classes=self.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=loss_cfg.get('no_object_weight', 0.1),
            losses=losses,
            num_points=loss_cfg.get('num_points', 12544),
            oversample_ratio=loss_cfg.get('oversample_ratio', 3.0),
            importance_sample_ratio=loss_cfg.get('importance_sample_ratio',
                                                 0.75),
            dn=dn,
            dn_losses=loss_cfg.get('dn_losses', losses),
            semantic_ce_loss=True)

    def forward(self, inputs, img_metas=None, targets=None):
        feats = self._transform_inputs(inputs)
        mask_features, _, multi_scale_features = self.pixel_decoder.forward_features(
            feats, masks=None)
        return self.predictor(multi_scale_features, mask_features,
                              targets=targets)

    def _semantic_targets(self, gt_semantic_seg, img_metas, device):
        targets = []
        for seg, meta in zip(gt_semantic_seg, img_metas):
            seg = seg.squeeze(0).to(device)
            labels = torch.unique(seg)
            labels = labels[(labels != self.ignore_index)
                            & (labels < self.num_classes)]
            masks = torch.stack([seg == label for label in labels],
                                dim=0).float() if labels.numel() > 0 else (
                                    seg.new_zeros((0, *seg.shape)).float())
            img_h, img_w = meta['img_shape'][:2]
            boxes_xyxy = box_ops.masks_to_boxes_bitmask(masks > 0)
            scale = boxes_xyxy.new_tensor([img_w, img_h, img_w, img_h])
            boxes = box_ops.box_xyxy_to_cxcywh(boxes_xyxy) / scale
            targets.append({
                'labels': labels.long(),
                'masks': masks,
                'boxes': boxes.clamp(min=0.0, max=1.0),
            })
        return targets

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg=None):
        del train_cfg
        targets = self._semantic_targets(gt_semantic_seg, img_metas,
                                         inputs[0].device)
        outputs, mask_dict = self(inputs, img_metas, targets)
        losses = self.criterion(outputs, targets, mask_dict)
        for key in list(losses.keys()):
            if key in self.criterion.weight_dict:
                losses[key] = losses[key] * self.criterion.weight_dict[key]
            else:
                losses.pop(key)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        del test_cfg
        outputs, _ = self(inputs, img_metas, targets=None)
        mask_cls_results = outputs['pred_logits']
        mask_pred_results = outputs['pred_masks']
        batch_h, batch_w = img_metas[0]['batch_input_shape']
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(batch_h, batch_w),
            mode='bilinear',
            align_corners=False)
        semseg = []
        for mask_cls, mask_pred, meta in zip(mask_cls_results,
                                             mask_pred_results, img_metas):
            img_h, img_w = meta['img_shape'][:2]
            mask_pred = mask_pred[:, :img_h, :img_w]
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
            mask_pred = mask_pred.sigmoid()
            semseg.append(torch.einsum('qc,qhw->chw', mask_cls, mask_pred))
        return torch.stack(semseg, dim=0)
