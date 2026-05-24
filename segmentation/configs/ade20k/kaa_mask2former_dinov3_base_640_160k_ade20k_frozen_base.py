_base_ = [
    '../_base_/models/mask2former_beit.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py',
]

crop_size = (640, 640)
pretrained = '/home/rliuar/0_Storage/hf_home/hub/dinov3-vitb16-pretrain-lvd1689m'
num_things_classes = 0
num_stuff_classes = 150
num_classes = num_things_classes + num_stuff_classes

model = dict(
    type='EncoderDecoderMask2Former',
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='KAAAdapterDINOv3Seg',
        pretrain_size=640,
        img_size=640,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        drop_path_rate=0.0,
        conv_inplane=64,
        interaction_indexes=[2, 5, 8, 11],
        out_indices=[2, 5, 8, 11],
        pretrained=pretrained,
        freeze_backbone=True,
        add_vit_feature=True,
        kaa_rank=64,
        kaa_gate_init=1e-3,
        kaa_variant='product',
        align_loss_weight=0.0,
        align_sample_size=1024,
    ),
    decode_head=dict(
        in_channels=[768, 768, 768, 768],
        feat_channels=768,
        out_channels=768,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        num_queries=100,
        pixel_decoder=dict(
            encoder=dict(
                transformerlayers=dict(
                    attn_cfgs=dict(embed_dims=768, num_heads=12),
                    ffn_cfgs=dict(embed_dims=768, feedforward_channels=3072))),
            positional_encoding=dict(num_feats=384)),
        positional_encoding=dict(num_feats=384),
        transformer_decoder=dict(
            transformerlayers=dict(
                attn_cfgs=dict(embed_dims=768, num_heads=12),
                ffn_cfgs=dict(embed_dims=768, feedforward_channels=3072),
                feedforward_channels=3072)),
        loss_cls=dict(class_weight=[1.0] * num_classes + [0.1]),
    ),
    train_cfg=dict(num_points=12544, oversample_ratio=3.0, importance_sample_ratio=0.75),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(426, 426)),
)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(2048, 640), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='ResizeToMultiple', size_divisor=32),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]),
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        data_root='/home/rliuar/0_Storage/scratch/datasets/ade20k/ADEChallengeData2016',
        pipeline=train_pipeline),
    val=dict(
        data_root='/home/rliuar/0_Storage/scratch/datasets/ade20k/ADEChallengeData2016',
        pipeline=test_pipeline),
    test=dict(
        data_root='/home/rliuar/0_Storage/scratch/datasets/ade20k/ADEChallengeData2016',
        pipeline=test_pipeline),
)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.60),
)
optimizer_config = dict(grad_clip=dict(max_norm=0.01, norm_type=2))
lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=8000, max_keep_ckpts=2)
evaluation = dict(interval=8000, metric='mIoU', pre_eval=True, save_best='mIoU')
fp16 = dict(loss_scale=dict(init_scale=512))
find_unused_parameters = True
work_dir = 'work_dirs/ade20k/kaa_mask2former_dinov3_base_640_160k_ade20k_frozen_product'
