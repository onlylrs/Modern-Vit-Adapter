import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast

from ops.modules import MSDeformAttn

from .position_encoding import PositionEmbeddingSine
from .utils import _get_activation_fn, _get_clones


def _norm(norm, channels):
    if norm in (None, ''):
        return nn.Identity()
    if norm == 'GN':
        return nn.GroupNorm(32, channels)
    raise ValueError(f'Unsupported norm: {norm}')


class MSDeformAttnTransformerEncoderOnly(nn.Module):
    def __init__(self,
                 d_model=256,
                 nhead=8,
                 num_encoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.1,
                 activation='relu',
                 num_feature_levels=4,
                 enc_n_points=4):
        super().__init__()
        encoder_layer = MSDeformAttnTransformerEncoderLayer(
            d_model, dim_feedforward, dropout, activation, num_feature_levels,
            nhead, enc_n_points)
        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer,
                                                      num_encoder_layers)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        nn.init.normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, h, w = mask.shape
        valid_h = torch.sum(~mask[:, :, 0], 1)
        valid_w = torch.sum(~mask[:, 0, :], 1)
        return torch.stack([valid_w.float() / w, valid_h.float() / h], -1)

    def forward(self, srcs, masks, pos_embeds):
        if masks is None:
            masks = [
                torch.zeros((x.size(0), x.size(2), x.size(3)),
                            device=x.device, dtype=torch.bool)
                for x in srcs
            ]

        src_flatten, mask_flatten, lvl_pos_embed_flatten = [], [], []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, _, h, w = src.shape
            spatial_shapes.append((h, w))
            src_flatten.append(src.flatten(2).transpose(1, 2))
            mask_flatten.append(mask.flatten(1))
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed_flatten.append(
                pos_embed + self.level_embed[lvl].view(1, 1, -1))
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long,
                                         device=src_flatten.device)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1, )),
            spatial_shapes.prod(1).cumsum(0)[:-1],
        ))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index,
                              valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        return memory, spatial_shapes, level_start_index


class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 d_ffn=1024,
                 dropout=0.1,
                 activation='relu',
                 n_levels=4,
                 n_heads=8,
                 n_points=4):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        return self.norm2(src)

    def forward(self, src, pos, reference_points, spatial_shapes,
                level_start_index, padding_mask=None):
        src2 = self.self_attn(
            self.with_pos_embed(src, pos), reference_points, src,
            spatial_shapes, level_start_index, padding_mask)
        src = self.norm1(src + self.dropout1(src2))
        return self.forward_ffn(src)


class MSDeformAttnTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            h_int, w_int = int(h.item()), int(w.item())
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, h_int - 0.5, h_int, dtype=torch.float32,
                               device=device),
                torch.linspace(0.5, w_int - 0.5, w_int, dtype=torch.float32,
                               device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 1] * h_int)
            ref_x = ref_x.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 0] * w_int)
            reference_points_list.append(torch.stack((ref_x, ref_y), -1))
        reference_points = torch.cat(reference_points_list, 1)
        return reference_points[:, :, None] * valid_ratios[:, None]

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios,
                pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=src.device)
        for layer in self.layers:
            output = layer(output, pos, reference_points, spatial_shapes,
                           level_start_index, padding_mask)
        return output


class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, norm='GN', activation=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, bias=(norm in (None, '')))
        self.norm = _norm(norm, out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.norm(self.conv(x))
        if self.activation is not None:
            x = self.activation(x)
        return x


class MaskDINOEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 strides=(4, 8, 16, 32),
                 transformer_in_indices=(1, 2, 3),
                 transformer_dropout=0.0,
                 transformer_nheads=8,
                 transformer_dim_feedforward=1024,
                 transformer_enc_layers=6,
                 conv_dim=256,
                 mask_dim=256,
                 norm='GN',
                 common_stride=4,
                 num_feature_levels=3,
                 total_num_feature_levels=4,
                 feature_order='low2high'):
        super().__init__()
        self.in_features = list(range(len(in_channels)))
        self.feature_strides = list(strides)
        self.feature_channels = list(in_channels)
        self.feature_order = feature_order
        self.transformer_in_indices = list(transformer_in_indices)
        self.maskdino_num_feature_levels = num_feature_levels
        self.total_num_feature_levels = total_num_feature_levels
        self.common_stride = common_stride

        transformer_channels = [in_channels[i] for i in self.transformer_in_indices]
        transformer_strides = [strides[i] for i in self.transformer_in_indices]
        self.transformer_num_feature_levels = len(self.transformer_in_indices)
        self.low_resolution_index = transformer_channels.index(
            max(transformer_channels))
        self.high_resolution_index = 0 if feature_order == 'low2high' else -1

        input_proj_list = []
        for ch in transformer_channels[::-1]:
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(ch, conv_dim, kernel_size=1),
                nn.GroupNorm(32, conv_dim),
            ))
        ch = max(transformer_channels)
        for _ in range(self.total_num_feature_levels -
                       self.transformer_num_feature_levels):
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(ch, conv_dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, conv_dim),
            ))
            ch = conv_dim
        self.input_proj = nn.ModuleList(input_proj_list)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            num_feature_levels=self.total_num_feature_levels,
        )
        self.pe_layer = PositionEmbeddingSine(conv_dim // 2, normalize=True)
        self.mask_features = nn.Conv2d(conv_dim, mask_dim, 1)
        nn.init.xavier_uniform_(self.mask_features.weight)
        nn.init.constant_(self.mask_features.bias, 0)

        stride = min(transformer_strides)
        self.num_fpn_levels = max(int(math.log2(stride) -
                                      math.log2(self.common_stride)), 1)
        lateral_convs, output_convs = [], []
        for in_ch in self.feature_channels[:self.num_fpn_levels]:
            lateral_convs.append(ConvNorm(in_ch, conv_dim, 1, norm=norm))
            output_convs.append(ConvNorm(conv_dim, conv_dim, 3, padding=1,
                                         norm=norm, activation=F.relu))
        self.lateral_convs = nn.ModuleList(lateral_convs[::-1])
        self.output_convs = nn.ModuleList(output_convs[::-1])

    @autocast(enabled=False)
    def forward_features(self, features, masks=None):
        features = [f.float() for f in features]
        srcs, pos = [], []
        extra_srcs, extra_pos = [], []

        if self.total_num_feature_levels > self.transformer_num_feature_levels:
            smallest = features[self.transformer_in_indices[self.low_resolution_index]]
            for level in range(self.transformer_num_feature_levels,
                               self.total_num_feature_levels):
                src = self.input_proj[level](
                    smallest if level == self.transformer_num_feature_levels
                    else extra_srcs[-1])
                extra_srcs.append(src)
                extra_pos.append(self.pe_layer(src))

        for idx, feat_idx in enumerate(self.transformer_in_indices[::-1]):
            x = features[feat_idx]
            srcs.append(self.input_proj[idx](x))
            pos.append(self.pe_layer(x))

        if self.feature_order == 'low2high':
            srcs.extend(extra_srcs[::-1])
            pos.extend(extra_pos[::-1])
        else:
            srcs = extra_srcs[::-1] + srcs
            pos = extra_pos[::-1] + pos

        y, spatial_shapes, level_start_index = self.transformer(srcs, masks, pos)
        bs = y.shape[0]
        split_sizes = []
        for i in range(self.total_num_feature_levels):
            if i < self.total_num_feature_levels - 1:
                split_sizes.append(
                    int((level_start_index[i + 1] - level_start_index[i]).item()))
            else:
                split_sizes.append(int((y.shape[1] - level_start_index[i]).item()))
        y = torch.split(y, split_sizes, dim=1)

        out = [
            z.transpose(1, 2).view(bs, -1, int(spatial_shapes[i][0].item()),
                                   int(spatial_shapes[i][1].item()))
            for i, z in enumerate(y)
        ]
        for idx, feat_idx in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
            x = features[feat_idx]
            cur_fpn = self.lateral_convs[idx](x)
            y = cur_fpn + F.interpolate(
                out[self.high_resolution_index], size=cur_fpn.shape[-2:],
                mode='bilinear', align_corners=False)
            out.append(self.output_convs[idx](y))

        multi_scale_features = out[:self.total_num_feature_levels]
        return self.mask_features(out[-1]), out[0], multi_scale_features
