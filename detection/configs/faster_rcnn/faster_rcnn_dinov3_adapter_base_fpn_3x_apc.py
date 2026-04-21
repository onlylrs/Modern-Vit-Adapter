# Copyright (c) Shanghai AI Lab. All rights reserved.
_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/apc_detection.py',
    '../_base_/schedules/schedule_3x.py',
    '../_base_/default_runtime.py',
]

pretrained = 'dinov3-vitb16-pretrain-lvd1689m'

model = dict(
    backbone=dict(
        _delete_=True,
        type='ViTAdapterDINOv3',
        pretrain_size=592,
        img_size=592,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        drop_path_rate=0.0,
        with_cp=True,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=12,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        pretrained=pretrained,
        freeze_backbone=True,
        adapter_mode='official_adapter',
    ),
    neck=dict(type='FPN', in_channels=[768, 768, 768, 768], out_channels=256, num_outs=5),
    roi_head=dict(bbox_head=dict(num_classes=6)),
)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.60),
)
optimizer_config = dict(grad_clip=None)

checkpoint_config = dict(interval=1, max_keep_ckpts=3, save_last=True)
