import copy
from typing import Optional

import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast

from ops.modules import MSDeformAttn

from . import box_ops
from .utils import (MLP, _get_activation_fn, _get_clones,
                    gen_encoder_output_proposals,
                    gen_sineembed_for_position, inverse_sigmoid)


class TransformerDecoder(nn.Module):
    def __init__(self,
                 decoder_layer,
                 num_layers,
                 norm=None,
                 return_intermediate=False,
                 d_model=256,
                 query_dim=4,
                 num_feature_levels=1,
                 dec_layer_share=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers,
                                  layer_share=dec_layer_share)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate, 'MaskDINO requires intermediate decoder outputs'
        assert query_dim in [2, 4]
        self.query_dim = query_dim
        self.num_feature_levels = num_feature_levels
        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        self.query_scale = None
        self.bbox_embed = None
        self.d_model = d_model
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def forward(self,
                tgt,
                memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None,
                level_start_index: Optional[Tensor] = None,
                spatial_shapes: Optional[Tensor] = None,
                valid_ratios: Optional[Tensor] = None,
                bbox_embed: Optional[nn.ModuleList] = None):
        output = tgt
        reference_points = refpoints_unsigmoid.sigmoid().to(tgt.device)
        intermediate = []
        ref_points = [reference_points]

        for layer_id, layer in enumerate(self.layers):
            reference_points_input = reference_points[:, :, None] * torch.cat(
                [valid_ratios, valid_ratios], -1)[None, :]
            query_sine_embed = gen_sineembed_for_position(
                reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            output = layer(
                tgt=output,
                tgt_query_pos=query_pos,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_reference_points=reference_points_input,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_level_start_index=level_start_index,
                memory_spatial_shapes=spatial_shapes,
                memory_pos=pos,
                self_attn_mask=tgt_mask,
                cross_attn_mask=memory_mask,
            )

            if bbox_embed is not None:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = bbox_embed[layer_id](output).to(tgt.device)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                reference_points = outputs_unsig.sigmoid().detach()
                ref_points.append(outputs_unsig.sigmoid())

            intermediate.append(self.norm(output))

        return [[out.transpose(0, 1) for out in intermediate],
                [ref.transpose(0, 1) for ref in ref_points]]


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 d_ffn=1024,
                 dropout=0.1,
                 activation='relu',
                 n_levels=4,
                 n_heads=8,
                 n_points=4):
        super().__init__()
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        return self.norm3(tgt)

    @autocast(enabled=False)
    def forward(self,
                tgt: Optional[Tensor],
                tgt_query_pos: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_reference_points: Optional[Tensor] = None,
                memory: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                memory_level_start_index: Optional[Tensor] = None,
                memory_spatial_shapes: Optional[Tensor] = None,
                memory_pos: Optional[Tensor] = None,
                self_attn_mask: Optional[Tensor] = None,
                cross_attn_mask: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, tgt_query_pos)
        tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
            tgt_reference_points.transpose(0, 1).contiguous(),
            memory.transpose(0, 1),
            memory_spatial_shapes,
            memory_level_start_index,
            memory_key_padding_mask,
        ).transpose(0, 1)
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        return self.forward_ffn(tgt)


class MaskDINODecoder(nn.Module):
    def __init__(self,
                 in_channels=256,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 nheads=8,
                 dim_feedforward=2048,
                 dec_layers=9,
                 mask_dim=256,
                 enforce_input_project=False,
                 two_stage=True,
                 dn='no',
                 noise_scale=0.4,
                 dn_num=100,
                 initialize_box_type='no',
                 initial_pred=True,
                 learn_tgt=False,
                 total_num_feature_levels=4,
                 dropout=0.0,
                 activation='relu',
                 dec_n_points=4,
                 return_intermediate_dec=True,
                 query_dim=4,
                 dec_layer_share=False,
                 semantic_ce_loss=False):
        super().__init__()
        self.num_feature_levels = total_num_feature_levels
        self.initial_pred = initial_pred
        self.dn = dn
        self.learn_tgt = learn_tgt
        self.noise_scale = noise_scale
        self.dn_num = dn_num
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.two_stage = two_stage
        self.initialize_box_type = initialize_box_type
        self.total_num_feature_levels = total_num_feature_levels
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.semantic_ce_loss = semantic_ce_loss

        if not two_stage or self.learn_tgt:
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
        if not two_stage and initialize_box_type == 'no':
            self.query_embed = nn.Embedding(num_queries, 4)
        if two_stage:
            self.enc_output = nn.Linear(hidden_dim, hidden_dim)
            self.enc_output_norm = nn.LayerNorm(hidden_dim)

        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, 1))
            else:
                self.input_proj.append(nn.Identity())

        cls_out_channels = num_classes + 1 if semantic_ce_loss else num_classes
        self.class_embed = nn.Linear(hidden_dim, cls_out_channels)
        self.label_enc = nn.Embedding(num_classes, hidden_dim)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        decoder_layer = DeformableTransformerDecoderLayer(
            hidden_dim, dim_feedforward, dropout, activation,
            self.num_feature_levels, nheads, dec_n_points)
        self.decoder = TransformerDecoder(
            decoder_layer,
            self.num_layers,
            self.decoder_norm,
            return_intermediate=return_intermediate_dec,
            d_model=hidden_dim,
            query_dim=query_dim,
            num_feature_levels=self.num_feature_levels,
            dec_layer_share=dec_layer_share,
        )
        self.hidden_dim = hidden_dim
        bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)
        # Official MaskDINO shares this module by repeating the same object in a
        # ModuleList. PyTorch 2.9 rejects duplicate parameters in optimizers, so
        # keep per-layer heads here for MMDet training compatibility.
        self.bbox_embed = nn.ModuleList(
            [copy.deepcopy(bbox_embed) for _ in range(self.num_layers)])

    def get_valid_ratio(self, mask):
        _, h, w = mask.shape
        valid_h = torch.sum(~mask[:, :, 0], 1)
        valid_w = torch.sum(~mask[:, 0, :], 1)
        return torch.stack([valid_w.float() / w, valid_h.float() / h], -1)

    def pred_box(self, reference, hs, ref0=None):
        outputs_coord_list = [] if ref0 is None else [ref0.to(reference[0].device)]
        for layer_ref_sig, layer_bbox_embed, layer_hs in zip(
                reference[:-1], self.bbox_embed, hs):
            layer_delta_unsig = layer_bbox_embed(layer_hs).to(reference[0].device)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            outputs_coord_list.append(layer_outputs_unsig.sigmoid())
        return torch.stack(outputs_coord_list)

    def prepare_for_dn(self, targets, tgt, refpoint_embed, batch_size):
        if not self.training or self.dn == 'no':
            return tgt, refpoint_embed, None, None

        device = tgt.device
        known_num = [int(t['labels'].numel()) for t in targets]
        max_known = max(known_num) if known_num else 0
        scalar = self.dn_num // max_known if max_known > 0 else 0
        if scalar == 0:
            return tgt, refpoint_embed, None, None

        labels = torch.cat([t['labels'] for t in targets])
        boxes = torch.cat([t['boxes'] for t in targets])
        batch_idx = torch.cat([
            torch.full_like(t['labels'].long(), i)
            for i, t in enumerate(targets)
        ])

        known_indice = torch.arange(labels.numel(), device=device).repeat(scalar)
        known_labels = labels.repeat(scalar)
        known_bid = batch_idx.repeat(scalar)
        known_bboxs = boxes.repeat(scalar, 1)
        known_labels_expand = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if self.noise_scale > 0:
            p = torch.rand_like(known_labels_expand.float())
            chosen = torch.nonzero(p < self.noise_scale * 0.5).view(-1)
            if chosen.numel() > 0:
                new_label = torch.randint(
                    low=0,
                    high=self.num_classes,
                    size=(chosen.numel(), ),
                    device=device,
                    dtype=known_labels_expand.dtype)
                known_labels_expand.scatter_(0, chosen, new_label)

            diff = torch.zeros_like(known_bbox_expand)
            diff[:, :2] = known_bbox_expand[:, 2:] / 2
            diff[:, 2:] = known_bbox_expand[:, 2:]
            known_bbox_expand += (
                torch.rand_like(known_bbox_expand) * 2 - 1.0) * diff * self.noise_scale
            known_bbox_expand = known_bbox_expand.clamp(min=0.0, max=1.0)

        input_label_embed = self.label_enc(known_labels_expand.long())
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)
        single_pad = max_known
        pad_size = single_pad * scalar
        padding_label = tgt.new_zeros((batch_size, pad_size, self.hidden_dim))
        padding_bbox = refpoint_embed.new_zeros((batch_size, pad_size, 4))
        input_query_label = torch.cat([padding_label, tgt], dim=1)
        input_query_bbox = torch.cat([padding_bbox, refpoint_embed], dim=1)

        map_known_indice = torch.cat([
            torch.arange(num, device=device) for num in known_num
        ])
        map_known_indice = torch.cat([
            map_known_indice + single_pad * i for i in range(scalar)
        ]).long()
        input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
        input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        tgt_size = pad_size + self.num_queries
        attn_mask = torch.zeros((tgt_size, tgt_size), dtype=torch.bool,
                                device=device)
        attn_mask[pad_size:, :pad_size] = True
        for i in range(scalar):
            start = single_pad * i
            end = single_pad * (i + 1)
            attn_mask[start:end, :start] = True
            attn_mask[start:end, end:pad_size] = True

        mask_dict = {
            'known_indice': known_indice.long(),
            'batch_idx': batch_idx.long(),
            'map_known_indice': map_known_indice.long(),
            'known_lbs_bboxes': (known_labels, known_bboxs),
            'pad_size': pad_size,
            'scalar': scalar,
        }
        return input_query_label, input_query_bbox, attn_mask, mask_dict

    def dn_post_process(self, outputs_class, outputs_coord, mask_dict,
                        outputs_mask):
        pad_size = mask_dict['pad_size']
        output_known_class = outputs_class[:, :, :pad_size, :]
        outputs_class = outputs_class[:, :, pad_size:, :]
        output_known_coord = outputs_coord[:, :, :pad_size, :]
        outputs_coord = outputs_coord[:, :, pad_size:, :]
        output_known_mask = outputs_mask[:, :, :pad_size, :]
        outputs_mask = outputs_mask[:, :, pad_size:, :]
        out = {
            'pred_logits': output_known_class[-1],
            'pred_boxes': output_known_coord[-1],
            'pred_masks': output_known_mask[-1],
            'aux_outputs': self._set_aux_loss(
                output_known_class, output_known_mask, output_known_coord),
        }
        mask_dict['output_known_lbs_bboxes'] = out
        return outputs_class, outputs_coord, outputs_mask

    def forward(self, x, mask_features, masks=None, targets=None):
        assert len(x) == self.num_feature_levels
        bs = x[0].shape[0]
        if masks is None:
            masks = [
                torch.zeros((src.size(0), src.size(2), src.size(3)),
                            device=src.device, dtype=torch.bool)
                for src in x
            ]

        src_flatten, mask_flatten, spatial_shapes = [], [], []
        for i in range(self.num_feature_levels):
            idx = self.num_feature_levels - 1 - i
            src = x[idx]
            spatial_shapes.append(src.shape[-2:])
            src_flatten.append(self.input_proj[idx](src).flatten(2).transpose(1, 2))
            mask_flatten.append(masks[idx].flatten(1))
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long,
                                         device=src_flatten.device)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1, )),
            spatial_shapes.prod(1).cumsum(0)[:-1],
        ))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        predictions_class = []
        predictions_mask = []
        if self.two_stage:
            output_memory, output_proposals = gen_encoder_output_proposals(
                src_flatten, mask_flatten, spatial_shapes)
            output_memory = self.enc_output_norm(self.enc_output(output_memory))
            enc_outputs_class_unselected = self.class_embed(output_memory)
            enc_outputs_coord_unselected = self.bbox_embed[0](
                output_memory) + output_proposals
            topk_proposals = torch.topk(
                enc_outputs_class_unselected.max(-1)[0], self.num_queries,
                dim=1)[1]
            refpoint_embed_undetach = torch.gather(
                enc_outputs_coord_unselected, 1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            refpoint_embed = refpoint_embed_undetach.detach()
            tgt_undetach = torch.gather(
                output_memory, 1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, self.hidden_dim))
            outputs_class, outputs_mask = self.forward_prediction_heads(
                tgt_undetach.transpose(0, 1), mask_features)
            tgt = tgt_undetach.detach()
            if self.learn_tgt:
                tgt = self.query_feat.weight[None].repeat(bs, 1, 1)
            interm_outputs = {
                'pred_logits': outputs_class,
                'pred_boxes': refpoint_embed_undetach.sigmoid(),
                'pred_masks': outputs_mask,
            }
            if self.initialize_box_type == 'mask2box':
                flat_mask = outputs_mask.detach().flatten(0, 1)
                h, w = outputs_mask.shape[-2:]
                refpoint_embed = box_ops.masks_to_boxes(flat_mask > 0)
                refpoint_embed = box_ops.box_xyxy_to_cxcywh(refpoint_embed)
                refpoint_embed = refpoint_embed / torch.as_tensor(
                    [w, h, w, h], dtype=torch.float, device=flat_mask.device)
                refpoint_embed = inverse_sigmoid(
                    refpoint_embed.reshape(outputs_mask.shape[0],
                                           outputs_mask.shape[1], 4))
            elif self.initialize_box_type == 'bitmask':
                flat_mask = outputs_mask.detach().flatten(0, 1)
                h, w = outputs_mask.shape[-2:]
                refpoint_embed = box_ops.masks_to_boxes_bitmask(flat_mask > 0)
                refpoint_embed = box_ops.box_xyxy_to_cxcywh(refpoint_embed)
                refpoint_embed = refpoint_embed / torch.as_tensor(
                    [w, h, w, h], dtype=torch.float, device=flat_mask.device)
                refpoint_embed = inverse_sigmoid(
                    refpoint_embed.reshape(outputs_mask.shape[0],
                                           outputs_mask.shape[1], 4))
            elif self.initialize_box_type != 'no':
                raise NotImplementedError(
                    'Only initialize_box_type no/mask2box/bitmask are supported.')
        else:
            tgt = self.query_feat.weight[None].repeat(bs, 1, 1)
            refpoint_embed = self.query_embed.weight[None].repeat(bs, 1, 1)
            interm_outputs = None

        tgt_mask = None
        mask_dict = None
        if self.dn != 'no' and self.training:
            assert targets is not None
            tgt, refpoint_embed, tgt_mask, mask_dict = self.prepare_for_dn(
                targets, tgt, refpoint_embed, bs)

        if self.initial_pred:
            outputs_class, outputs_mask = self.forward_prediction_heads(
                tgt.transpose(0, 1), mask_features, pred_mask=True)
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=src_flatten.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=None,
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=tgt_mask,
            bbox_embed=self.bbox_embed,
        )
        for i, output in enumerate(hs):
            outputs_class, outputs_mask = self.forward_prediction_heads(
                output.transpose(0, 1), mask_features,
                pred_mask=self.training or i == len(hs) - 1)
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        out_boxes = self.pred_box(references, hs, refpoint_embed.sigmoid())
        if mask_dict is not None:
            predictions_mask = torch.stack(predictions_mask)
            predictions_class = torch.stack(predictions_class)
            predictions_class, out_boxes, predictions_mask = self.dn_post_process(
                predictions_class, out_boxes, mask_dict, predictions_mask)
            predictions_class = list(predictions_class)
            predictions_mask = list(predictions_mask)
        if self.training:
            predictions_class[-1] = (
                predictions_class[-1] + 0.0 * self.label_enc.weight.sum())
        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'pred_boxes': out_boxes[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class, predictions_mask, out_boxes),
        }
        if self.two_stage:
            out['interm_outputs'] = interm_outputs
        return out, mask_dict

    def forward_prediction_heads(self, output, mask_features, pred_mask=True):
        decoder_output = self.decoder_norm(output).transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        outputs_mask = None
        if pred_mask:
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum('bqc,bchw->bqhw', mask_embed,
                                        mask_features)
        return outputs_class, outputs_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, out_boxes):
        return [
            {'pred_logits': a, 'pred_masks': b, 'pred_boxes': c}
            for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1],
                               out_boxes[:-1])
        ]
