from __future__ import annotations

import logging
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import BACKBONES
from ops.modules import MSDeformAttn
from timm.models.layers import trunc_normal_
from torch.nn.init import normal_

from integrations.dinov3_hf_backbone import OfficialDINOv3Backbone, normalize_interaction_indexes

from .adapter_modules import Extractor, Injector, SpatialPriorModule, deform_inputs

_logger = logging.getLogger(__name__)


class InteractionBlockWithCls(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=6,
        n_points=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop=0.0,
        drop_path=0.0,
        with_cffn=True,
        cffn_ratio=0.25,
        init_values=0.0,
        deform_ratio=1.0,
        extra_extractor=False,
        with_cp=False,
    ):
        super().__init__()
        self.extractor = Extractor(
            dim=dim,
            n_levels=1,
            num_heads=num_heads,
            n_points=n_points,
            norm_layer=norm_layer,
            deform_ratio=deform_ratio,
            with_cffn=with_cffn,
            cffn_ratio=cffn_ratio,
            drop=drop,
            drop_path=drop_path,
            with_cp=with_cp,
        )
        if extra_extractor:
            self.extra_extractors = nn.Sequential(
                *[
                    Extractor(
                        dim=dim,
                        num_heads=num_heads,
                        n_points=n_points,
                        norm_layer=norm_layer,
                        with_cffn=with_cffn,
                        cffn_ratio=cffn_ratio,
                        deform_ratio=deform_ratio,
                        drop=drop,
                        drop_path=drop_path,
                        with_cp=with_cp,
                    )
                    for _ in range(2)
                ]
            )
        else:
            self.extra_extractors = None

    def forward(self, x, c, cls, deform_inputs1, deform_inputs2, h_c, w_c, h_toks, w_toks):
        del cls, deform_inputs1, h_toks, w_toks
        c = self.extractor(
            query=c,
            reference_points=deform_inputs2[0],
            feat=x,
            spatial_shapes=deform_inputs2[1],
            level_start_index=deform_inputs2[2],
            H=h_c,
            W=w_c,
        )
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(
                    query=c,
                    reference_points=deform_inputs2[0],
                    feat=x,
                    spatial_shapes=deform_inputs2[1],
                    level_start_index=deform_inputs2[2],
                    H=h_c,
                    W=w_c,
                )
        return x, c


class LocalInteractionBlockWithCls(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=6,
        n_points=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop=0.0,
        drop_path=0.0,
        with_cffn=True,
        cffn_ratio=0.25,
        init_values=0.0,
        deform_ratio=1.0,
        extra_extractor=False,
        with_cp=False,
    ):
        super().__init__()
        self.injector = Injector(
            dim=dim,
            n_levels=3,
            num_heads=num_heads,
            init_values=init_values,
            n_points=n_points,
            norm_layer=norm_layer,
            deform_ratio=deform_ratio,
            with_cp=with_cp,
        )
        self.extractor = Extractor(
            dim=dim,
            n_levels=1,
            num_heads=num_heads,
            n_points=n_points,
            norm_layer=norm_layer,
            deform_ratio=deform_ratio,
            with_cffn=with_cffn,
            cffn_ratio=cffn_ratio,
            drop=drop,
            drop_path=drop_path,
            with_cp=with_cp,
        )
        if extra_extractor:
            self.extra_extractors = nn.Sequential(
                *[
                    Extractor(
                        dim=dim,
                        num_heads=num_heads,
                        n_points=n_points,
                        norm_layer=norm_layer,
                        with_cffn=with_cffn,
                        cffn_ratio=cffn_ratio,
                        deform_ratio=deform_ratio,
                        drop=drop,
                        drop_path=drop_path,
                        with_cp=with_cp,
                    )
                    for _ in range(2)
                ]
            )
        else:
            self.extra_extractors = None

    def forward(self, x, c, cls, deform_inputs1, deform_inputs2, h_c, w_c, h_toks, w_toks):
        del cls, h_toks, w_toks
        x = self.injector(
            query=x,
            reference_points=deform_inputs1[0],
            feat=c,
            spatial_shapes=deform_inputs1[1],
            level_start_index=deform_inputs1[2],
        )
        c = self.extractor(
            query=c,
            reference_points=deform_inputs2[0],
            feat=x,
            spatial_shapes=deform_inputs2[1],
            level_start_index=deform_inputs2[2],
            H=h_c,
            W=w_c,
        )
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(
                    query=c,
                    reference_points=deform_inputs2[0],
                    feat=x,
                    spatial_shapes=deform_inputs2[1],
                    level_start_index=deform_inputs2[2],
                    H=h_c,
                    W=w_c,
                )
        return x, c


@BACKBONES.register_module()
class ViTAdapterDINOv3Seg(nn.Module):
    def __init__(
        self,
        pretrain_size=224,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=6,
        init_values=0.0,
        interaction_indexes=None,
        with_cffn=True,
        cffn_ratio=0.25,
        deform_ratio=1.0,
        add_vit_feature=True,
        use_extra_extractor=True,
        pretrained=None,
        freeze_backbone=True,
        adapter_mode="official_adapter",
        with_cp=True,
        **kwargs,
    ):
        super().__init__()
        if pretrained is None:
            raise ValueError("pretrained must point to a DINOv3 checkpoint root")

        resolved_root = self._resolve_checkpoint_root(pretrained)
        self.backbone = OfficialDINOv3Backbone.from_checkpoint(resolved_root)
        self.freeze_backbone = bool(freeze_backbone)
        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = normalize_interaction_indexes(interaction_indexes or [])
        self.add_vit_feature = add_vit_feature
        self.adapter_mode = adapter_mode

        block_cls = self._interaction_block_cls(adapter_mode)

        embed_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_size

        expected_patch_size = kwargs.get("patch_size")
        if expected_patch_size is not None and int(expected_patch_size) != self.patch_size:
            raise ValueError(f"Checkpoint patch size {self.patch_size} does not match config patch_size={expected_patch_size}")

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim)
        self.interactions = nn.Sequential(
            *[
                block_cls(
                    dim=embed_dim,
                    num_heads=deform_num_heads,
                    n_points=n_points,
                    init_values=init_values,
                    drop_path=0.0,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio,
                    deform_ratio=deform_ratio,
                    extra_extractor=((i == len(self.interaction_indexes) - 1) and use_extra_extractor),
                    with_cp=with_cp,
                )
                for i in range(len(self.interaction_indexes))
            ]
        )

        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

        _logger.info("Loaded official-style DINOv3 backbone from %s", resolved_root)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, (2.0 / fan_out) ** 0.5)
            if m.bias is not None:
                m.bias.data.zero_()

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    @staticmethod
    def _resolve_checkpoint_root(checkpoint_root):
        root = Path(checkpoint_root)
        if root.is_absolute() or root.exists():
            return root
        for parent in Path(__file__).resolve().parents:
            candidate = parent / root
            if candidate.exists():
                return candidate
        return root

    @staticmethod
    def _interaction_block_cls(adapter_mode):
        if adapter_mode == "official_adapter":
            return InteractionBlockWithCls
        if adapter_mode == "local_interaction":
            return LocalInteractionBlockWithCls
        raise ValueError(f"Unsupported adapter_mode: {adapter_mode}")

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        h_c, w_c = x.shape[2] // 16, x.shape[3] // 16
        h_toks, w_toks = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size

        backbone_ctx = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        with backbone_ctx:
            all_layers = self.backbone.get_intermediate_layers(
                x,
                n=self.interaction_indexes,
                return_class_token=True,
            )

        x_for_shape, _ = all_layers[0]
        bs, _, dim = x_for_shape.shape
        del x_for_shape

        outs = []
        for i, layer in enumerate(self.interactions):
            patch_tokens, cls_token = all_layers[i]
            patch_tokens, c = layer(
                patch_tokens,
                c,
                cls_token,
                deform_inputs1,
                deform_inputs2,
                h_c,
                w_c,
                h_toks,
                w_toks,
            )
            outs.append(patch_tokens.transpose(1, 2).view(bs, dim, h_toks, w_toks).contiguous())

        c2_len, c3_len, c4_len = c2.size(1), c3.size(1), c4.size(1)
        c2 = c[:, 0:c2_len, :]
        c3 = c[:, c2_len:c2_len + c3_len, :]
        c4 = c[:, c2_len + c3_len:c2_len + c3_len + c4_len, :]

        c2 = c2.transpose(1, 2).view(bs, dim, h_c * 2, w_c * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, h_c, w_c).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, h_c // 2, w_c // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, size=(4 * h_c, 4 * w_c), mode="bilinear", align_corners=False)
            x2 = F.interpolate(x2, size=(2 * h_c, 2 * w_c), mode="bilinear", align_corners=False)
            x3 = F.interpolate(x3, size=(h_c, w_c), mode="bilinear", align_corners=False)
            x4 = F.interpolate(x4, size=(h_c // 2, w_c // 2), mode="bilinear", align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]


DINOv3SegAdapter = ViTAdapterDINOv3Seg
