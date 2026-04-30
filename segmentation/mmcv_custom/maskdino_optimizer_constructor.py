import copy
import json
import warnings

import torch
from mmcv.runner import (OPTIMIZER_BUILDERS, DefaultOptimizerConstructor,
                         get_dist_info)


@OPTIMIZER_BUILDERS.register_module()
class MaskDINOOptimizerConstructor(DefaultOptimizerConstructor):
    """Detectron2-style MaskDINO AdamW parameter grouping."""

    def add_params(self, params, module, prefix='', is_dcn_module=None):
        del prefix, is_dcn_module
        backbone_multiplier = self.paramwise_cfg.get('backbone_multiplier', 0.1)
        weight_decay_norm = self.paramwise_cfg.get('weight_decay_norm', 0.0)
        weight_decay_embed = self.paramwise_cfg.get('weight_decay_embed', 0.0)
        bypass_duplicate = self.paramwise_cfg.get('bypass_duplicate', True)
        display = self.paramwise_cfg.get('display', False)

        defaults = dict(lr=self.base_lr, weight_decay=self.base_wd)
        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        memo = set()
        debug_groups = []
        for module_name, sub_module in module.named_modules():
            for param_name, value in sub_module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                full_name = f'{module_name}.{param_name}' if module_name else param_name
                if value in memo:
                    if bypass_duplicate:
                        warnings.warn(
                            f'{full_name} is duplicate. It is skipped since '
                            'bypass_duplicate=True')
                        continue
                    raise ValueError(f'{full_name} appears in multiple groups')
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if 'backbone' in module_name:
                    hyperparams['lr'] = hyperparams['lr'] * backbone_multiplier
                if ('relative_position_bias_table' in param_name
                        or 'absolute_pos_embed' in param_name):
                    hyperparams['weight_decay'] = 0.0
                if isinstance(sub_module, norm_module_types):
                    hyperparams['weight_decay'] = weight_decay_norm
                if isinstance(sub_module, torch.nn.Embedding):
                    hyperparams['weight_decay'] = weight_decay_embed

                params.append({'params': [value], **hyperparams})
                if display:
                    debug_groups.append({
                        'param_name': full_name,
                        'lr': hyperparams['lr'],
                        'weight_decay': hyperparams['weight_decay'],
                    })

        if display:
            rank, _ = get_dist_info()
            if rank == 0:
                print('MaskDINO param groups = %s' %
                      json.dumps(debug_groups, indent=2))
