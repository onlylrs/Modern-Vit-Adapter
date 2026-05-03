# Inspired by ViT-Split's DINOv2 split-fusion backbone.
# This is a DINOv3 adaptation for this repository, using
# integrations.OfficialDINOv3Backbone instead of torch.hub DINOv2.

from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import DeformConv2d
from mmcv.runner import BaseModule
from mmseg.models.builder import BACKBONES

from integrations.dinov3_hf_backbone import OfficialDINOv3Backbone


def _uniform_indices(num_layers: int, count: int, start: int = 0) -> list[int]:
    if count <= 0:
        return []
    if count == 1:
        return [num_layers - 1]
    step = (num_layers - 1 - start) / float(count - 1)
    return [int(round(start + idx * step)) for idx in range(count)]


class DeformableConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.offsets = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size,
                                 kernel_size=kernel_size, padding=padding)
        self.deform_conv = DeformConv2d(in_channels, out_channels,
                                        kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        return self.deform_conv(x, self.offsets(x))


@BACKBONES.register_module()
class DINOv3SplitFusionSeg(BaseModule):
    """ViT-Split style frozen-feature + split-head fusion for DINOv3."""

    def __init__(
        self,
        pretrained,
        out_indices='uniform',
        out_indices_count=4,
        out_indices_start=2,
        select_layers=None,
        channels=None,
        tuning_type='frozen',
        *args,
        **kwargs,
    ):
        super().__init__()
        self.backbone = OfficialDINOv3Backbone.from_checkpoint(self._resolve_checkpoint_root(pretrained))
        self.patch_size = self.backbone.patch_size
        self.channels = channels or self.backbone.embed_dim
        num_layers = self.backbone.n_blocks

        if out_indices == 'uniform':
            self.out_indices = _uniform_indices(num_layers, out_indices_count, out_indices_start)
        else:
            self.out_indices = list(out_indices)
        if select_layers is None:
            self.select_layers = [num_layers - 2, num_layers - 1]
        elif select_layers == 'uniform':
            self.select_layers = _uniform_indices(num_layers, 2, max(0, num_layers - 4))
        else:
            self.select_layers = list(select_layers)

        self.split_head = nn.ModuleList([copy.deepcopy(self.backbone.layers[idx]) for idx in self.select_layers])
        self._set_backbone_trainability(tuning_type)

        self.frozen_conv = nn.Sequential(
            nn.Conv2d(self.channels * len(self.out_indices), self.channels, kernel_size=1),
            nn.GELU(),
            DeformableConvNet(self.channels, self.channels),
            nn.GELU(),
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.channels * 2, self.channels, kernel_size=1),
            nn.GELU(),
            DeformableConvNet(self.channels, self.channels),
            nn.GELU(),
        )

    def _set_backbone_trainability(self, tuning_type):
        if tuning_type == 'frozen':
            self.backbone.requires_grad_(False)
            self.backbone.eval()
        elif tuning_type == 'all':
            self.backbone.requires_grad_(True)
        elif isinstance(tuning_type, list):
            self.backbone.requires_grad_(False)
            for idx in tuning_type:
                self.backbone.layers[idx].requires_grad_(True)
        else:
            raise AttributeError(f'{tuning_type} is not supported')

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

    def train(self, mode=True):
        super().train(mode)
        if not any(param.requires_grad for param in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def _tokens_to_feature(self, hidden_states, x):
        patch_tokens, _, _ = self.backbone.split_tokens(hidden_states, norm=True)
        b, _, h, w = x.shape
        return patch_tokens.reshape(
            b, h // self.patch_size, w // self.patch_size, -1
        ).permute(0, 3, 1, 2).contiguous()

    def _forward_split_features(self, x):
        hidden_states, position_embeddings = self.backbone.prepare_tokens(x)
        out_indices = set(self.out_indices)
        split_start = self.select_layers[0]
        split_input = hidden_states.detach() if split_start == 0 else None
        frozen_outputs = {}

        stop_idx = max(max(out_indices), split_start - 1)
        backbone_ctx = torch.no_grad() if not any(p.requires_grad for p in self.backbone.parameters()) else torch.enable_grad()
        with backbone_ctx:
            for idx, layer in enumerate(self.backbone.layers):
                hidden_states = layer(hidden_states, position_embeddings=position_embeddings)
                if idx in out_indices:
                    frozen_outputs[idx] = hidden_states
                if idx == split_start - 1:
                    split_input = hidden_states.detach()
                if idx >= stop_idx:
                    break

        if split_input is None:
            split_input = hidden_states.detach()
        split_states = split_input
        for layer in self.split_head:
            split_states = layer(split_states, position_embeddings=position_embeddings)

        frozen_features = tuple(self._tokens_to_feature(frozen_outputs[idx], x) for idx in self.out_indices)
        tuned_features = self._tokens_to_feature(split_states, x)
        return frozen_features, tuned_features

    def forward(self, x):
        frozen_features, tuned_features = self._forward_split_features(x)
        frozen_features = self.frozen_conv(torch.cat(frozen_features, dim=1))
        return self.fusion_conv(torch.cat([frozen_features, tuned_features], dim=1))
