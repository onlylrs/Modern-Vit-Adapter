# Copyright (c) Bosch and ActionLab. All rights reserved.
#
# Source: ViT-Split, segmentation/mmseg_custom/models/backbones/vit_dinov2_split.py
# Adapted for this repository with explicit source comments and optional
# uniform layer selection. Core DINOv2 Split-Fusion modules follow ViT-Split.

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
from mmseg.models.builder import BACKBONES


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
    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
    return torch.hub.load(repo_or_dir='facebookresearch/dinov2', model=backbone_name)


@BACKBONES.register_module()
class DINOViTSplitFusion(BaseModule):
    """DINOv2 ViT-Split Fusion backbone.

    Source: ViT-Split. This keeps the frozen DINOv2 branch, copied split_head,
    feature hook, deformable fusion, and single feature output used by its
    segmentation linear heads.
    """

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
        *args,
        **kwargs,
    ):
        super().__init__()
        self.output_orgimg = output_orgimg
        self.register_version = register_version

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

        backbone_model.forward = partial(backbone_model.get_intermediate_layers, n=out_indices, reshape=True)
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
        self.frozen_conv = nn.Sequential(
            nn.Conv2d(channels * len(out_indices), channels, kernel_size=1, padding=0),
            nn.GELU(),
            DeformableConvNet(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, padding=0),
            nn.GELU(),
            DeformableConvNet(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.bn_norm = nn.SyncBatchNorm(channels)

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
        return self.fusion_conv(x)


@BACKBONES.register_module()
class DINOViTSplitLearnableGate(BaseModule):
    """DINOv2 ViT-Split LearnableGate backbone from ViT-Split."""

    def __init__(
        self,
        backbone_size='small',
        register_version=False,
        tune_register=True,
        channels=384,
        out_num=4,
        select_layers=None,
        tuning_type='frozen',
        output_orgimg=False,
        freeze_learnable_gate=False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.output_orgimg = output_orgimg
        self.register_version = register_version

        backbone_model = _load_dinov2(backbone_size, register_version)
        if register_version:
            self.num_register_tokens = backbone_model.num_register_tokens

        num_layers = len(backbone_model.blocks)
        if select_layers is None:
            select_layers = [num_layers - 2, num_layers - 1]
        elif select_layers == 'uniform':
            select_layers = _uniform_indices(num_layers, count=2, start=max(0, num_layers - 4))
        self.select_layers = list(select_layers)

        self.split_head = nn.Sequential(*[copy.deepcopy(backbone_model.blocks[layer_id])
                                          for layer_id in self.select_layers])
        self.split_activations = None
        backbone_model.blocks[self.select_layers[0] - 1].register_forward_hook(self.get_activation)
        backbone_model.forward = partial(
            backbone_model.get_intermediate_layers,
            n=[i for i in range(num_layers)],
            reshape=True,
            norm=True,
            return_class_token=False,
        )
        self.learnable_gate = LearnableGate(n=num_layers, k=out_num, out_num=out_num, temperature=0.5)
        self.backbone = backbone_model
        self.out_num = out_num

        self.frozen_conv = nn.Sequential(
            nn.Conv2d(channels * out_num, channels, kernel_size=1, padding=0),
            nn.GELU(),
            DeformableConvNet(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.fusion_net = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, padding=0),
            nn.GELU(),
            DeformableConvNet(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
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

        if freeze_learnable_gate:
            for param in self.learnable_gate.parameters():
                param.requires_grad = False
        if register_version and tune_register:
            for param in backbone_model.register_tokens:
                param.requires_grad = True

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
        return self.fusion_net(x)
