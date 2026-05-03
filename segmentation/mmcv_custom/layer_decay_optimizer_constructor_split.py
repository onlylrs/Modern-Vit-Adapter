# Copyright (c) ByteDance, Inc. and its affiliates.
#
# Source: ViT-Split, segmentation/mmcv_custom/layer_decay_optimizer_constructor_split.py
# Adapted only to document provenance in this repository.

import json
import re

from mmcv.runner import OPTIMIZER_BUILDERS, DefaultOptimizerConstructor, get_dist_info


def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ('backbone.split_head.cls_token', 'backbone.split_head.mask_token',
                    'backbone.split_head.pos_embed', 'backbone.split_head.visual_embed'):
        return 0
    if var_name.startswith('backbone.split_head.visual_embed'):
        return 0
    if var_name.startswith('backbone.split_head.patch_embed'):
        return 0
    if var_name.startswith('backbone.split_head.'):
        try:
            layer_id = int(re.findall(r'\d+', var_name)[0])
        except Exception:
            return 0
        return layer_id + 1
    return num_max_layer - 1


@OPTIMIZER_BUILDERS.register_module()
class LayerDecayOptimizerConstructorSplit(DefaultOptimizerConstructor):
    def add_params(self, params, module, prefix='', is_dcn_module=None):
        parameter_groups = {}
        print(self.paramwise_cfg)
        num_layers = self.paramwise_cfg.get('num_layers') + 2
        layer_decay_rate = self.paramwise_cfg.get('layer_decay_rate')
        print('Build LayerDecayOptimizerConstructor %f - %d' %
              (layer_decay_rate, num_layers))
        weight_decay = self.base_wd

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith('.bias') or name in (
                    'pos_embed', 'backbone.split_head.cls_token',
                    'backbone.split_head.visual_embed'):
                group_name = 'no_decay'
                this_weight_decay = 0.
            else:
                group_name = 'decay'
                this_weight_decay = weight_decay

            layer_id = get_num_layer_for_vit(name, num_layers)
            group_name = 'layer_%d_%s' % (layer_id, group_name)

            if group_name not in parameter_groups:
                scale = layer_decay_rate ** (num_layers - layer_id - 1)
                parameter_groups[group_name] = {
                    'weight_decay': this_weight_decay,
                    'params': [],
                    'param_names': [],
                    'lr_scale': scale,
                    'group_name': group_name,
                    'lr': scale * self.base_lr,
                }

            parameter_groups[group_name]['params'].append(param)
            parameter_groups[group_name]['param_names'].append(name)
        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    'param_names': parameter_groups[key]['param_names'],
                    'lr_scale': parameter_groups[key]['lr_scale'],
                    'lr': parameter_groups[key]['lr'],
                    'weight_decay': parameter_groups[key]['weight_decay'],
                }
            print('Param groups = %s' % json.dumps(to_display, indent=2))

        params.extend(parameter_groups.values())
