from __future__ import annotations

import logging
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import BACKBONES
from ops.modules import MSDeformAttn
from timm.layers import trunc_normal_
from torch.nn.init import normal_

from integrations.dinov3_hf_backbone import (
    OfficialDINOv3Backbone,
    normalize_interaction_indexes,
    normalize_interaction_ranges,
)

from .comer_modules import CNN, CTIBlock, deform_inputs, deform_inputs_only_one

_logger = logging.getLogger(__name__)


class DINOv3CTIBlock(CTIBlock):
    """CoMer CTI block variant that runs official DINOv3 blocks with prefix tokens."""

    def forward(
        self,
        x,
        c,
        prefix_tokens,
        blocks,
        position_embeddings,
        deform_inputs1,
        deform_inputs2,
        H,
        W,
    ):
        del deform_inputs1
        deform_inputs = deform_inputs_only_one(x, H * 16, W * 16)
        if self.use_CTI_toV:
            c = self.mrfp(c, H, W)
            c_select1 = c[:, :H * W * 4, :]
            c_select2 = c[:, H * W * 4:H * W * 4 + H * W, :]
            c_select3 = c[:, H * W * 4 + H * W:, :]
            c = torch.cat([c_select1, c_select2 + x, c_select3], dim=1)
            x = self.cti_tov(
                query=x,
                reference_points=deform_inputs[0],
                feat=c,
                spatial_shapes=deform_inputs[1],
                level_start_index=deform_inputs[2],
                H=H,
                W=W,
            )

        prefix_len = prefix_tokens.shape[1]
        hidden_states = torch.cat((prefix_tokens, x), dim=1)
        for block in blocks:
            hidden_states = block(hidden_states, position_embeddings=position_embeddings)
        prefix_tokens = hidden_states[:, :prefix_len, :]
        x = hidden_states[:, prefix_len:, :]

        if self.use_CTI_toC:
            c = self.cti_toc(
                query=c,
                reference_points=deform_inputs2[0],
                feat=x,
                spatial_shapes=deform_inputs2[1],
                level_start_index=deform_inputs2[2],
                H=H,
                W=W,
            )
        if self.extra_CTIs is not None:
            for cti in self.extra_CTIs:
                c = cti(
                    query=c,
                    reference_points=deform_inputs2[0],
                    feat=x,
                    spatial_shapes=deform_inputs2[1],
                    level_start_index=deform_inputs2[2],
                    H=H,
                    W=W,
                )
        return x, c, prefix_tokens


@BACKBONES.register_module()
class ViTCoMerDINOv3(nn.Module):
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
        use_CTI_toV=None,
        use_CTI_toC=None,
        cnn_feature_interaction=None,
        dim_ratio=0.5,
        pretrained=None,
        freeze_backbone=True,
        with_cp=True,
        **kwargs,
    ):
        super().__init__()
        if pretrained is None:
            raise ValueError("pretrained must point to a DINOv3 checkpoint root")

        resolved_root = self._resolve_checkpoint_root(pretrained)
        self.backbone = OfficialDINOv3Backbone.from_checkpoint(resolved_root)
        self._freeze_unused_dinov3_params()
        self.freeze_backbone = bool(freeze_backbone)
        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = normalize_interaction_indexes(interaction_indexes or [])
        self.interaction_ranges = normalize_interaction_ranges(interaction_indexes or [])
        if self.interaction_indexes and max(self.interaction_indexes) >= self.backbone.n_blocks:
            raise ValueError(
                f"interaction_indexes must be within DINOv3 depth {self.backbone.n_blocks}: "
                f"{interaction_indexes}"
            )

        embed_dim = self.backbone.embed_dim
        self.embed_dim = embed_dim
        self.patch_size = self.backbone.patch_size
        self.add_vit_feature = add_vit_feature

        expected_patch_size = kwargs.get("patch_size")
        if expected_patch_size is not None and int(expected_patch_size) != self.patch_size:
            raise ValueError(f"Checkpoint patch size {self.patch_size} does not match config patch_size={expected_patch_size}")
        if self.patch_size != 16:
            raise ValueError(f"ViT-CoMer assumes stride-16 patch tokens, got DINOv3 patch size {self.patch_size}")
        expected_embed_dim = kwargs.get("embed_dim")
        if expected_embed_dim is not None and int(expected_embed_dim) != embed_dim:
            raise ValueError(f"Checkpoint embed dim {embed_dim} does not match config embed_dim={expected_embed_dim}")
        expected_depth = kwargs.get("depth")
        if expected_depth is not None and int(expected_depth) != self.backbone.n_blocks:
            raise ValueError(f"Checkpoint depth {self.backbone.n_blocks} does not match config depth={expected_depth}")

        interaction_count = len(self.interaction_ranges)
        if interaction_count == 0:
            raise ValueError("interaction_indexes must define at least one ViT-CoMer interaction range")
        use_CTI_toV = self._normalize_flags(use_CTI_toV, interaction_count, True)
        use_CTI_toC = self._normalize_flags(use_CTI_toC, interaction_count, True)
        cnn_feature_interaction = self._normalize_flags(cnn_feature_interaction, interaction_count, False)

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = CNN(inplanes=conv_inplane, embed_dim=embed_dim)
        self.interactions = nn.Sequential(
            *[
                DINOv3CTIBlock(
                    dim=embed_dim,
                    num_heads=deform_num_heads,
                    n_points=n_points,
                    init_values=init_values,
                    drop_path=0.0,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio,
                    deform_ratio=deform_ratio,
                    extra_CTI=(idx == interaction_count - 1),
                    with_cp=with_cp,
                    use_CTI_toV=use_CTI_toV[idx],
                    use_CTI_toC=use_CTI_toC[idx],
                    dim_ratio=dim_ratio,
                    cnn_feature_interaction=cnn_feature_interaction[idx],
                    extra_num=4,
                )
                for idx in range(interaction_count)
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

        _logger.info("Loaded DINOv3 ViT-CoMer backbone from %s", resolved_root)

    @staticmethod
    def _normalize_flags(value, expected_len, default):
        if value is None:
            return [default] * expected_len
        if isinstance(value, bool):
            return [value] * expected_len
        flags = list(value)
        if len(flags) != expected_len:
            raise ValueError(f"Expected {expected_len} flags, got {len(flags)}")
        return [bool(flag) for flag in flags]

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

    def _freeze_unused_dinov3_params(self):
        # ViT-CoMer consumes prepared prefix+patch tokens directly and never uses the
        # masked-image token or the final backbone norm module in its loss path.
        # Leaving these trainable causes DDP to error on unused parameters.
        unused_params = [
            getattr(self.backbone.model.embeddings, "mask_token", None),
            getattr(self.backbone.model.norm, "weight", None),
            getattr(self.backbone.model.norm, "bias", None),
        ]
        for param in unused_params:
            if isinstance(param, nn.Parameter):
                param.requires_grad_(False)

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

        hidden_states, position_embeddings = self.backbone.prepare_tokens(x)
        bs, _, dim = hidden_states.shape
        prefix_len = self.backbone.num_prefix_tokens
        next_layer = 0

        for (start, end), layer in zip(self.interaction_ranges, self.interactions):
            if start > next_layer:
                hidden_states = self.backbone.run_layers(hidden_states, position_embeddings, next_layer, start - 1)
            prefix_tokens = hidden_states[:, :prefix_len, :]
            patch_tokens = hidden_states[:, prefix_len:, :]
            blocks = self.backbone.layers[start:end + 1]
            patch_tokens, c, prefix_tokens = layer(
                patch_tokens,
                c,
                prefix_tokens,
                blocks,
                position_embeddings,
                deform_inputs1,
                deform_inputs2,
                h_c,
                w_c,
            )
            hidden_states = torch.cat((prefix_tokens, patch_tokens), dim=1)
            next_layer = end + 1

        c2_len, c3_len, c4_len = c2.size(1), c3.size(1), c4.size(1)
        c2 = c[:, 0:c2_len, :]
        c3 = c[:, c2_len:c2_len + c3_len, :]
        c4 = c[:, c2_len + c3_len:c2_len + c3_len + c4_len, :]

        c2 = c2.transpose(1, 2).view(bs, dim, h_c * 2, w_c * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, h_c, w_c).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, h_c // 2, w_c // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x3 = patch_tokens.transpose(1, 2).view(bs, dim, h_toks, w_toks).contiguous()
            x1 = F.interpolate(x3, size=(4 * h_c, 4 * w_c), mode='bilinear', align_corners=False)
            x2 = F.interpolate(x3, size=(2 * h_c, 2 * w_c), mode='bilinear', align_corners=False)
            x4 = F.interpolate(x3, size=(h_c // 2, w_c // 2), mode='bilinear', align_corners=False)
            x3 = F.interpolate(x3, size=(h_c, w_c), mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]


DINOv3CoMer = ViTCoMerDINOv3
