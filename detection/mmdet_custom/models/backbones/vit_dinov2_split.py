# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Source: ViT-Split, detection/mmdet_custom/models/backbones/vit_dinov2_split.py
# Adapted for this repository with explicit source comments and optional
# uniform frozen-layer selection. Core DINOv2 Split-Fusion modules follow
# ViT-Split; this port uses torch.hub DINOv2 loading to avoid vendoring the
# full ViT-Split dinov2 subtree.

from __future__ import annotations

import copy
import itertools
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import DeformConv2d
from mmcv.runner import BaseModule
from mmdet.models.builder import BACKBONES


def _uniform_indices(num_layers: int, count: int, start: int = 0) -> list[int]:
    if count <= 0:
        return []
    if count == 1:
        return [num_layers - 1]
    step = (num_layers - 1 - start) / float(count - 1)
    return [int(round(start + idx * step)) for idx in range(count)]


def _resolve_indices(indices, num_layers: int, *, count: int, start: int = 0):
    if indices == 'uniform':
        return _uniform_indices(num_layers, count=count, start=start)
    return list(indices)


class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        return F.pad(x, pads)


class DeformableConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.offsets = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size,
                                 kernel_size=kernel_size, padding=padding)
        self.deform_conv = DeformConv2d(in_channels, out_channels,
                                        kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        offsets = self.offsets(x)
        return self.deform_conv(x, offsets)


class LearnableGate(torch.nn.Module):
    """Source: ViT-Split LearnableGate."""

    def __init__(self, n, k, out_num, temperature=0.5):
        super().__init__()
        self.n = n
        self.k = k
        self.out_num = out_num
        self.temperature = temperature
        self.scores = torch.nn.Parameter(torch.randn(n, out_num))
        torch.nn.init.uniform_(self.scores, a=0, b=1)

    def forward(self, X):
        B, n, _, _, _ = X.shape
        assert n == self.n, 'Input feature dimension n must match the initialized n'
        scores = self.scores.unsqueeze(0).expand(B, -1, -1)
        soft_scores = F.softmax(scores / self.temperature, dim=1)
        topk_indices = torch.topk(soft_scores, self.k, dim=1).indices
        sparse_scores = torch.zeros_like(soft_scores)
        batch_idx = torch.arange(B, device=X.device).view(B, 1, 1).expand(B, self.k, self.out_num)
        channel_idx = torch.arange(self.out_num, device=X.device).view(1, 1, self.out_num).expand(
            B, self.k, self.out_num)
        sparse_scores[batch_idx, topk_indices, channel_idx] = soft_scores[batch_idx, topk_indices, channel_idx]
        gates = sparse_scores - scores.detach() + scores
        gates = gates / gates.sum(dim=1, keepdim=True)
        return gates


def _load_dinov2(backbone_size: str, register_version: bool):
    backbone_archs = {
        False: dict(small='vits14', base='vitb14', large='vitl14', giant='vitg14'),
        True: dict(small='vits14_reg', base='vitb14_reg', large='vitl14_reg', giant='vitg14_reg'),
    }
    backbone_arch = backbone_archs[register_version][backbone_size]
    backbone_name = f'dinov2_{backbone_arch}'
    validate_repo = getattr(torch.hub, '_validate_not_a_forked_repo', None)
    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
    try:
        return torch.hub.load(repo_or_dir='facebookresearch/dinov2', model=backbone_name)
    finally:
        if validate_repo is None:
            del torch.hub._validate_not_a_forked_repo
        else:
            torch.hub._validate_not_a_forked_repo = validate_repo


@BACKBONES.register_module()
class DINOViTSplitFusion(BaseModule):
    """DINOv2 ViT-Split Fusion backbone for FPN-style detection."""

    def __init__(
        self,
        backbone_size='small',
        register_version=False,
        tune_register=True,
        out_indices='uniform',
        out_indices_count=4,
        out_indices_start=2,
        select_layers=None,
        channels=384,
        tuning_type='frozen',
        output_orgimg=False,
        drop_path_rate=0,
        drop_path_uniform=False,
        *args,
        **kwargs,
    ):
        super().__init__()
        del drop_path_rate, drop_path_uniform
        self.output_orgimg = output_orgimg
        self.register_version = register_version
        self.channels = channels

        backbone_model = _load_dinov2(backbone_size, register_version)
        if register_version:
            self.num_register_tokens = backbone_model.num_register_tokens

        num_layers = len(backbone_model.blocks)
        out_indices = _resolve_indices(out_indices, num_layers, count=out_indices_count, start=out_indices_start)
        if select_layers is None:
            select_layers = [num_layers - 2, num_layers - 1]
        elif select_layers == 'uniform':
            select_layers = _uniform_indices(num_layers, count=2, start=max(0, num_layers - 4))
        self.select_layers = list(select_layers)

        self.split_head = nn.Sequential(*[copy.deepcopy(backbone_model.blocks[layer_id])
                                          for layer_id in self.select_layers])
        self.split_activations = None
        backbone_model.blocks[self.select_layers[0] - 1].register_forward_hook(self.get_activation)
        backbone_model.forward = partial(backbone_model.get_intermediate_layers, n=out_indices,
                                         reshape=True, norm=True)
        self.patch_size = backbone_model.patch_embed.patch_size[0]
        backbone_model.register_forward_pre_hook(lambda _, x: CenterPadding(self.patch_size)(x[0]))

        if tuning_type == 'frozen':
            for param in backbone_model.parameters():
                param.requires_grad = False
        elif tuning_type == 'all':
            for param in backbone_model.parameters():
                param.requires_grad = True
        elif isinstance(tuning_type, list):
            for param in backbone_model.parameters():
                param.requires_grad = False
            for layer_id in tuning_type:
                for param in backbone_model.blocks[layer_id].parameters():
                    param.requires_grad = True
        else:
            raise AttributeError(f'{tuning_type} is not supported')

        if register_version and tune_register:
            for param in backbone_model.register_tokens:
                param.requires_grad = True

        self.backbone = backbone_model
        frozen_out_dim = channels
        self.frozen_conv = nn.Sequential(
            nn.Conv2d(channels * len(out_indices), channels, kernel_size=1, padding=0),
            nn.GELU(),
            DeformableConvNet(channels, frozen_out_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.fusion_net = nn.Sequential(
            nn.Conv2d(channels + frozen_out_dim, channels, kernel_size=1, padding=0),
            nn.GELU(),
            DeformableConvNet(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, 2, 2),
            nn.GroupNorm(32, channels),
            nn.GELU(),
            nn.ConvTranspose2d(channels, channels, 2, 2),
            nn.GroupNorm(32, channels),
            nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, 2, 2),
            nn.GroupNorm(32, channels),
            nn.GELU(),
        )
        self.up3 = nn.Identity()
        self.up4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.norm1 = nn.SyncBatchNorm(channels)
        self.norm2 = nn.SyncBatchNorm(channels)
        self.norm3 = nn.SyncBatchNorm(channels)
        self.norm4 = nn.SyncBatchNorm(channels)

    @property
    def get_activation(self):
        def hook(model, input, output):
            self.split_activations = output.detach()
        return hook

    def reshape_vit_tokens(self, x, norm=True):
        b, _, _ = x.shape
        if norm:
            x = self.backbone.norm(x)
        x = x[:, 1:, :]
        if self.register_version:
            x = x[:, self.num_register_tokens:, :]
        return x.reshape(b, self.h, self.w, -1).permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        frozen_features = self.backbone(x)
        _, _, H, W = x.shape
        self.h, self.w = math.ceil(H / self.patch_size), math.ceil(W / self.patch_size)
        tuned_features = self.split_head(self.split_activations)
        tuned_features = self.reshape_vit_tokens(tuned_features)
        frozen_features = torch.cat(frozen_features, dim=1)
        frozen_features = self.frozen_conv(frozen_features)
        x = torch.cat([frozen_features, tuned_features], dim=1)
        x = self.fusion_net(x)
        x1 = self.norm1(self.up1(x))
        x2 = self.norm2(self.up2(x))
        x3 = self.norm3(self.up3(x))
        x4 = self.norm4(self.up4(x))
        return [x1, x2, x3, x4]


@BACKBONES.register_module()
class DINOViTSplitLearnableGate(DINOViTSplitFusion):
    """LearnableGate variant from ViT-Split, returning FPN-style features."""

    def __init__(self, *args, out_num=4, freeze_learnable_gate=False, **kwargs):
        super().__init__(*args, out_indices_count=out_num, **kwargs)
        num_layers = len(self.backbone.blocks)
        self.out_num = out_num
        self.backbone.forward = partial(
            self.backbone.get_intermediate_layers,
            n=[i for i in range(num_layers)],
            reshape=True,
            norm=True,
            return_class_token=False,
        )
        self.learnable_gate = LearnableGate(n=num_layers, k=out_num, out_num=out_num, temperature=0.5)
        self.frozen_conv[0] = nn.Conv2d(self.channels * out_num, self.channels, kernel_size=1, padding=0)
        if freeze_learnable_gate:
            for param in self.learnable_gate.parameters():
                param.requires_grad = False

    def forward(self, x):
        frozen_features_list = self.backbone(x)
        frozen_features = torch.stack(frozen_features_list, dim=1)
        gates = self.learnable_gate(frozen_features)
        frozen_features = torch.einsum('bldhw,blk->bkdhw', frozen_features, gates)
        b, k, d, h, w = frozen_features.shape
        frozen_features = frozen_features.reshape(b, k * d, h, w)
        frozen_features = self.frozen_conv(frozen_features)

        _, _, H, W = x.shape
        self.h, self.w = math.ceil(H / self.patch_size), math.ceil(W / self.patch_size)
        tuned_features = self.split_head(self.split_activations)
        tuned_features = self.reshape_vit_tokens(tuned_features)
        x = torch.cat([frozen_features, tuned_features], dim=1)
        x = self.fusion_net(x)
        return [self.norm1(self.up1(x)), self.norm2(self.up2(x)),
                self.norm3(self.up3(x)), self.norm4(self.up4(x))]
