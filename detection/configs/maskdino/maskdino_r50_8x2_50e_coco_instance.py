_base_ = [
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Resize',
        img_scale=(1024, 1024),
        ratio_range=(0.1, 2.0),
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=(1024, 1024),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(1024, 1024)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
data = dict(samples_per_gpu=2, workers_per_gpu=4,
            train=dict(pipeline=train_pipeline))

model = dict(
    type='MaskDINO',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/path/to/resnet50-0676ba61.pth')),
    bbox_head=dict(
        type='MaskDINOHead',
        in_channels=[256, 512, 1024, 2048],
        strides=[4, 8, 16, 32],
        num_classes=80,
        feat_channels=256,
        mask_dim=256,
        num_queries=300,
        pixel_decoder=dict(
            transformer_in_indices=(3, 2, 1),
            transformer_dropout=0.0,
            transformer_nheads=8,
            transformer_dim_feedforward=2048,
            transformer_enc_layers=6,
            common_stride=4,
            num_feature_levels=3,
            total_num_feature_levels=4,
            feature_order='low2high'),
        transformer_decoder=dict(
            nheads=8,
            dim_feedforward=2048,
            dec_layers=9,
            enforce_input_project=False,
            two_stage=True,
            dn='seg',
            noise_scale=0.4,
            dn_num=100,
            initialize_box_type='mask2box',
            initial_pred=True,
            learn_tgt=False,
            total_num_feature_levels=4,
            dropout=0.0,
            dec_n_points=4),
        loss_cfg=dict(
            losses=['labels', 'masks', 'boxes'],
            dn_losses=['labels', 'masks', 'boxes'],
            class_weight=4.0,
            mask_weight=5.0,
            dice_weight=5.0,
            box_weight=5.0,
            giou_weight=2.0,
            cost_class=4.0,
            cost_mask=5.0,
            cost_dice=5.0,
            cost_box=5.0,
            cost_giou=2.0,
            no_object_weight=0.1,
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75)),
    train_cfg=dict(),
    test_cfg=dict(max_per_image=100, object_mask_thr=0.25, focus_on_box=False))

optimizer = dict(
    _delete_=True,
    constructor='MaskDINOOptimizerConstructor',
    type='AdamW',
    lr=0.0001,
    weight_decay=0.05,
    paramwise_cfg=dict(
        backbone_multiplier=0.1,
        weight_decay_norm=0.0,
        weight_decay_embed=0.0,
        bypass_duplicate=True,
        display=False))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=0.01, norm_type=2))
lr_config = dict(
    _delete_=True,
    policy='step',
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=1.0,
    step=[44, 48])
runner = dict(type='EpochBasedRunner', max_epochs=50)
fp16 = dict(loss_scale=dict(init_scale=512))
checkpoint_config = dict(interval=1, max_keep_ckpts=1, save_last=True)
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
# load_from = '/path/to/maskdino_r50_50ep_mmdet_instance_converted.pth' See readme to download it.

