# Copyright (c) Shanghai AI Lab. All rights reserved.
_base_ = ['./maskdino_r50_8x2_50e_coco_instance.py']

pretrained = '/path/to/deit_base_patch16_224-b5f2ef4d.pth'

model = dict(
    backbone=dict(
        _delete_=True,
        type='ViTAdapter',
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        drop_path_rate=0.3,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=12,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        window_attn=[
            True, True, False, True, True, False,
            True, True, False, True, True, False
        ],
        window_size=[
            14, 14, None, 14, 14, None,
            14, 14, None, 14, 14, None
        ],
        pretrained=pretrained),
    # MaskDINO has its own pixel decoder, so do not insert an FPN neck here.
    neck=None,
    bbox_head=dict(
        in_channels=[768, 768, 768, 768],
        pixel_decoder=dict(
            transformer_in_indices=(3, 2, 1),
            transformer_dim_feedforward=2048,
            total_num_feature_levels=4,
            feature_order='low2high'),
        transformer_decoder=dict(total_num_feature_levels=4)))

# Converted from the official MaskDINO R50 COCO instance checkpoint. The
# ResNet-specific weights will be skipped by shape/key mismatch; decoder and
# mask/bbox heads can still initialize from it.
load_from = '/path/to/maskdino_r50_50ep_mmdet_instance_converted.pth'

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'level_embed': dict(decay_mult=0.),
            'pos_embed': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'bias': dict(decay_mult=0.)
        }))
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=0.01, norm_type=2))

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        ann_file='/path/to/coco/annotations/instances_train2017.json',
        img_prefix='/path/to/coco/train2017/'),
    val=dict(
        ann_file='/path/to/coco/annotations/instances_val2017.json',
        img_prefix='/path/to/coco/val2017/'),
    test=dict(
        ann_file='/path/to/coco/annotations/instances_val2017.json',
        img_prefix='/path/to/coco/val2017/'))

work_dir = '/path/to/work_dirs/maskdino_vit_adapter_base_50e_coco_instance'
