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


@HEADS.register_module()
class MultiScaleConvUpsampleLinearBNHead(BaseDecodeHead):
    """Using multiscale concatenation and deconvolution for img tokens with batchnorm."""

    def __init__(self, resize_factors=None, **kwargs):
        super().__init__(**kwargs)
        assert self.in_channels == self.channels
        self.bn = nn.SyncBatchNorm(self.in_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(self.channels * 4, self.channels, kernel_size=1, padding=0),
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
        x = torch.cat(inputs, dim=1)
        x = self.conv(x)
        x = self.up(x)
        return self.bn(x)

    def _transform_inputs(self, inputs):
        if self.input_transform == 'resize_concat':
            input_list = []
            for x in inputs:
                if isinstance(x, list):
                    input_list.extend(x)
                else:
                    input_list.append(x)
            inputs = input_list
            for i, x in enumerate(inputs):
                if len(x.shape) == 2:
                    inputs[i] = x[:, :, None, None]
            inputs = [inputs[i] for i in self.in_index]
            if self.resize_factors is not None:
                assert len(self.resize_factors) == len(inputs)
                inputs = [
                    resize(input=x, scale_factor=f, mode='bilinear' if f >= 1 else 'area')
                    for x, f in zip(inputs, self.resize_factors)
                ]
            upsampled_inputs = [
                resize(input=x, size=inputs[0].shape[2:], mode='bilinear',
                       align_corners=self.align_corners)
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]
        return inputs

    def forward(self, inputs):
        output = self._forward_feature(inputs)
        return self.cls_seg(output)


@HEADS.register_module()
class ConvUpsampleLinearBNHead(BaseDecodeHead):
    """Using deconvolution for img tokens with batchnorm."""

    def __init__(self, resize_factors=None, **kwargs):
        super().__init__(**kwargs)
        assert self.in_channels == self.channels
        self.bn = nn.SyncBatchNorm(self.in_channels)
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
        x = self.up(inputs)
        return self.bn(x)

    def forward(self, inputs):
        output = self._forward_feature(inputs)
        return self.cls_seg(output)


@HEADS.register_module()
class LinearBNHead(BaseDecodeHead):
    """Linear segmentation head with batchnorm from ViT-Split."""

    def __init__(self, resize_factors=None, **kwargs):
        super().__init__(**kwargs)
        assert self.in_channels == self.channels
        self.bn = nn.SyncBatchNorm(self.in_channels)
        self.resize_factors = resize_factors

    def _forward_feature(self, inputs):
        if isinstance(inputs, list):
            x = inputs[-1]
        else:
            x = inputs
        return self.bn(x)

    def forward(self, inputs):
        output = self._forward_feature(inputs)
        return self.cls_seg(output)


@HEADS.register_module()
class DconvUpsamplingBNHead(BaseDecodeHead):
    """Using Dconv upsampling img tokens and batchnorm."""

    def __init__(self, resize_factors=None, **kwargs):
        super().__init__(**kwargs)
        assert self.in_channels == self.channels
        self.bn = nn.SyncBatchNorm(self.in_channels)
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
        x = self.up(inputs)
        return self.bn(x)

    def _transform_inputs(self, inputs):
        if self.input_transform == 'resize_concat':
            input_list = []
            for x in inputs:
                if isinstance(x, list):
                    input_list.extend(x)
                else:
                    input_list.append(x)
            inputs = input_list
            for i, x in enumerate(inputs):
                if len(x.shape) == 2:
                    inputs[i] = x[:, :, None, None]
            inputs = [inputs[i] for i in self.in_index]
            if self.resize_factors is not None:
                assert len(self.resize_factors) == len(inputs)
                inputs = [
                    resize(input=x, scale_factor=f, mode='bilinear' if f >= 1 else 'area')
                    for x, f in zip(inputs, self.resize_factors)
                ]
            upsampled_inputs = [
                resize(input=x, size=inputs[0].shape[2:], mode='bilinear',
                       align_corners=self.align_corners)
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]
        return inputs

    def forward(self, inputs):
        output = self._forward_feature(inputs)
        return self.cls_seg(output)
