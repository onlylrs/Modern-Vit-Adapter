# Copyright (c) Bosch and ActionLab. All rights reserved.
#
# Source: ViT-Split, segmentation/mmseg_custom/models/decode_heads/linear_head.py
# Adapted for this repository by adding the missing mmseg resize import while
# keeping the ViT-Split linear/deconvolution heads as standalone options.

import torch
import torch.nn as nn
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize


def _assert_single_scale_channels(in_channels, channels):
    if isinstance(in_channels, (list, tuple)):
        assert all(ch == channels for ch in in_channels)
    else:
        assert in_channels == channels


class ViTSplitInputMixin:
    """Input selection helpers that also accept ViT-Split's single-tensor output."""

    def _flatten_inputs(self, inputs):
        if torch.is_tensor(inputs):
            return inputs

        flattened = []
        for x in inputs:
            if isinstance(x, (list, tuple)):
                flattened.extend(x)
            else:
                flattened.append(x)
        return flattened

    def _select_inputs(self, inputs):
        inputs = self._flatten_inputs(inputs)
        if torch.is_tensor(inputs):
            return inputs

        if self.input_transform == 'resize_concat':
            inputs = [self._as_4d(inputs[i]) for i in self.in_index]
            if self.resize_factors is not None:
                assert len(self.resize_factors) == len(inputs)
                inputs = [
                    resize(input=x, scale_factor=f, mode='bilinear' if f >= 1 else 'area')
                    for x, f in zip(inputs, self.resize_factors)
                ]
            inputs = [
                resize(input=x, size=inputs[0].shape[2:], mode='bilinear',
                       align_corners=self.align_corners)
                for x in inputs
            ]
            return torch.cat(inputs, dim=1)

        if self.input_transform == 'multiple_select':
            return [self._as_4d(inputs[i]) for i in self.in_index]

        return self._as_4d(inputs[self.in_index])

    @staticmethod
    def _as_4d(x):
        if x.dim() == 2:
            return x[:, :, None, None]
        return x

    def _concat_inputs(self, inputs):
        inputs = self._select_inputs(inputs)
        if isinstance(inputs, (list, tuple)):
            return torch.cat(inputs, dim=1)
        return inputs

    def _select_last_input(self, inputs):
        inputs = self._select_inputs(inputs)
        if isinstance(inputs, (list, tuple)):
            return inputs[-1]
        return inputs


@HEADS.register_module()
class MultiScaleConvUpsampleLinearBNHead(ViTSplitInputMixin, BaseDecodeHead):
    """Using multiscale concatenation and deconvolution for img tokens with batchnorm."""

    def __init__(self, resize_factors=None, **kwargs):
        super().__init__(**kwargs)
        if isinstance(self.in_channels, (list, tuple)):
            conv_in_channels = sum(self.in_channels)
        else:
            conv_in_channels = self.in_channels
        self.bn = nn.SyncBatchNorm(self.channels)
        self.conv = nn.Sequential(
            nn.Conv2d(conv_in_channels, self.channels, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(self.channels, self.channels, 2, 2),
            nn.GroupNorm(32, self.channels),
            nn.GELU(),
            nn.ConvTranspose2d(self.channels, self.channels, 2, 2),
        )
        self.resize_factors = resize_factors

    def _forward_feature(self, inputs):
        x = self._concat_inputs(inputs)
        x = self.conv(x)
        x = self.up(x)
        return self.bn(x)

    def forward(self, inputs):
        output = self._forward_feature(inputs)
        return self.cls_seg(output)


@HEADS.register_module()
class ConvUpsampleLinearBNHead(ViTSplitInputMixin, BaseDecodeHead):
    """Using deconvolution for img tokens with batchnorm."""

    def __init__(self, resize_factors=None, **kwargs):
        super().__init__(**kwargs)
        _assert_single_scale_channels(self.in_channels, self.channels)
        self.bn = nn.SyncBatchNorm(self.channels)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(self.channels, self.channels, 2, 2),
            nn.GroupNorm(32, self.channels),
            nn.GELU(),
            nn.ConvTranspose2d(self.channels, self.channels, 2, 2),
            nn.GroupNorm(32, self.channels),
            nn.GELU(),
        )
        self.resize_factors = resize_factors

    def _forward_feature(self, inputs):
        inputs = self._select_last_input(inputs)
        x = self.up(inputs)
        return self.bn(x)

    def forward(self, inputs):
        output = self._forward_feature(inputs)
        return self.cls_seg(output)


@HEADS.register_module()
class LinearBNHead(ViTSplitInputMixin, BaseDecodeHead):
    """Linear segmentation head with batchnorm from ViT-Split."""

    def __init__(self, resize_factors=None, **kwargs):
        super().__init__(**kwargs)
        _assert_single_scale_channels(self.in_channels, self.channels)
        self.bn = nn.SyncBatchNorm(self.channels)
        self.resize_factors = resize_factors

    def _forward_feature(self, inputs):
        x = self._select_last_input(inputs)
        return self.bn(x)

    def forward(self, inputs):
        output = self._forward_feature(inputs)
        return self.cls_seg(output)


@HEADS.register_module()
class DconvUpsamplingBNHead(ViTSplitInputMixin, BaseDecodeHead):
    """Using Dconv upsampling img tokens and batchnorm."""

    def __init__(self, resize_factors=None, **kwargs):
        super().__init__(**kwargs)
        _assert_single_scale_channels(self.in_channels, self.channels)
        self.bn = nn.SyncBatchNorm(self.channels)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(self.channels, self.channels, 2, 2),
            nn.GroupNorm(32, self.channels),
            nn.GELU(),
            nn.ConvTranspose2d(self.channels, self.channels, 2, 2),
            nn.GroupNorm(32, self.channels),
            nn.GELU(),
        )
        self.resize_factors = resize_factors

    def _forward_feature(self, inputs):
        inputs = self._select_last_input(inputs)
        x = self.up(inputs)
        return self.bn(x)

    def forward(self, inputs):
        output = self._forward_feature(inputs)
        return self.cls_seg(output)
