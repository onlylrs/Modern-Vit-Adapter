"""CROWN ViT-L/16 with ViT-Adapter spatial interactions for MMDetection."""

from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from mmdet.models.builder import BACKBONES
from ops.modules import MSDeformAttn
from torch.nn.init import normal_

from integrations.crown_loader import load_crown

from .adapter_modules import Extractor, Injector, SpatialPriorModule, deform_inputs


class CrownInteraction(nn.Module):
    def __init__(self, dim, num_heads, n_points, init_values, cffn_ratio,
                 deform_ratio, extra_extractor, with_cp):
        super().__init__()
        norm = partial(nn.LayerNorm, eps=1e-6)
        self.injector = Injector(
            dim=dim, n_levels=3, num_heads=num_heads, n_points=n_points,
            init_values=init_values, deform_ratio=deform_ratio,
            norm_layer=norm, with_cp=with_cp)
        def extractor():
            return Extractor(
                dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                deform_ratio=deform_ratio, cffn_ratio=cffn_ratio,
                norm_layer=norm, with_cp=with_cp)
        self.extractor = extractor()
        self.extra_extractors = nn.ModuleList([extractor() for _ in range(2)]) if extra_extractor else nn.ModuleList()
        self.with_cp = with_cp

    def forward(self, tokens, prior, blocks, inputs1, inputs2, h, w):
        # Keep the class token inside every official transformer block.
        patch = self.injector(tokens[:, 1:], inputs1[0], prior, inputs1[1], inputs1[2])
        tokens = torch.cat((tokens[:, :1], patch), dim=1)
        for block in blocks:
            tokens = checkpoint(block, tokens, use_reentrant=False) if self.with_cp and tokens.requires_grad else block(tokens)
        patch = tokens[:, 1:]
        for extractor in (self.extractor, *self.extra_extractors):
            prior = extractor(prior, inputs2[0], patch, inputs2[1], inputs2[2], h, w)
        return tokens, prior


@BACKBONES.register_module()
class ViTAdapterCROWN(nn.Module):
    def __init__(self, pretrained, official_root, interaction_indexes=((0, 5), (6, 11), (12, 17), (18, 23)),
                 conv_inplane=64, n_points=4, deform_num_heads=16, init_values=0.0,
                 cffn_ratio=0.25, deform_ratio=0.5, add_vit_feature=True,
                 use_extra_extractor=True, freeze_backbone=False, with_cp=True):
        super().__init__()
        self.backbone = load_crown(pretrained, official_root)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self.backbone.requires_grad_(False)
        self.interaction_indexes = [tuple(x) for x in interaction_indexes]
        if len(self.interaction_indexes) != 4 or any(
            a < 0 or b >= 24 or a > b for a, b in self.interaction_indexes
        ) or any(self.interaction_indexes[i][1] >= self.interaction_indexes[i + 1][0] for i in range(3)):
            raise ValueError("CROWN needs four ordered, nonoverlapping interaction ranges within 0..23")
        self.add_vit_feature = add_vit_feature
        self.with_cp = with_cp
        dim = 1024
        self.level_embed = nn.Parameter(torch.zeros(3, dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=dim)
        self.interactions = nn.ModuleList([
            CrownInteraction(dim, deform_num_heads, n_points, init_values, cffn_ratio,
                             deform_ratio, i == 3 and use_extra_extractor, with_cp)
            for i in range(4)
        ])
        self.up = nn.ConvTranspose2d(dim, dim, 2, 2)
        self.norms = nn.ModuleList([nn.SyncBatchNorm(dim) for _ in range(4)])
        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @staticmethod
    def _init_deform_weights(m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, image):
        if image.shape[-2] % 32 or image.shape[-1] % 32:
            raise ValueError("CROWN ViT-Adapter input must be padded to a multiple of 32")
        h, w = image.shape[-2] // 16, image.shape[-1] // 16
        inputs1, inputs2 = deform_inputs(image)
        c1, c2, c3, c4 = self.spm(image)
        c2, c3, c4 = (c2 + self.level_embed[0], c3 + self.level_embed[1], c4 + self.level_embed[2])
        prior = torch.cat((c2, c3, c4), dim=1)
        tokens = self.backbone.prepare_tokens_with_masks(image)
        blocks = [block for chunk in self.backbone.blocks for block in chunk if not isinstance(block, nn.Identity)]
        outputs = []
        next_block = 0
        for (start, end), interaction in zip(self.interaction_indexes, self.interactions):
            for block in blocks[next_block:start]:
                tokens = checkpoint(block, tokens, use_reentrant=False) if self.with_cp and tokens.requires_grad else block(tokens)
            tokens, prior = interaction(tokens, prior, blocks[start:end + 1], inputs1, inputs2, h, w)
            outputs.append(tokens[:, 1:].transpose(1, 2).reshape(image.shape[0], 1024, h, w))
            next_block = end + 1
        sizes = [c2.shape[1], c3.shape[1], c4.shape[1]]
        p2, p3, p4 = prior.split(sizes, dim=1)
        p2 = p2.transpose(1, 2).reshape(image.shape[0], 1024, h * 2, w * 2)
        p3 = p3.transpose(1, 2).reshape(image.shape[0], 1024, h, w)
        p4 = p4.transpose(1, 2).reshape(image.shape[0], 1024, h // 2, w // 2)
        pyramid = [self.up(p2) + c1, p2, p3, p4]
        if self.add_vit_feature:
            sizes = [(h * 4, w * 4), (h * 2, w * 2), (h, w), (h // 2, w // 2)]
            pyramid = [p + F.interpolate(v, size=s, mode="bilinear", align_corners=False)
                       for p, v, s in zip(pyramid, outputs, sizes)]
        return [norm(p) for norm, p in zip(self.norms, pyramid)]
