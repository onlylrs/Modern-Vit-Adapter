# Source: ViT-Split, segmentation/configs/ade20k/vitsplit/linear_dinov2_vitsplit_base_512_20k_ade20k.py
# Adapted for this repository: frozen DINOv2 layers use explicit uniform
# sampling, while the split head keeps the official final 3 consecutive layers.

dataset_type = 'ADE20KDataset'
data_root = 'data/ade/ADEChallengeData2016'
total_iters = 40000
initial_lr = 2e-4
wd = 1e-2
lr_mult = 0.1
crop_size = (512, 512)
vit_hidden_dims = 768
num_classes = 150
pergpu_batch_size = 2
total_layers = 12


def uniform_indices(num_layers, count, start=0):
    if count == 1:
        return [num_layers - 1]
    gap = (num_layers - 1 - start) / float(count - 1)
    return [int(round(start + idx * gap)) for idx in range(count)]


frozen_indices = uniform_indices(total_layers, 4, start=2)
tuned_indices = [9, 10, 11]
register_version = False
tune_register = False

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(crop_size[0] * 4, crop_size[0]), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='ToMask'),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(crop_size[0] * 4, crop_size[0]),
        img_ratios=1.0,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]),
]
data = dict(
    samples_per_gpu=pergpu_batch_size,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/training',
        ann_dir='annotations/training',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline))

log_config = dict(interval=50, hooks=[dict(type='PrintLrGroupHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
find_unused_parameters = True
norm_cfg = dict(type='SyncBN', requires_grad=True)

optimizer = dict(
    type='AdamW',
    lr=initial_lr,
    weight_decay=wd,
    paramwise_cfg=dict(custom_keys={'backbone.split_head': dict(lr_mult=lr_mult)}),
    betas=(0.9, 0.999))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=0.9,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=total_iters)
checkpoint_config = dict(by_epoch=False, interval=10000)
evaluation = dict(interval=10000, metric='mIoU', pre_eval=True)
fp16 = None

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='DINOViTSplitFusion',
        backbone_size='base',
        register_version=register_version,
        tune_register=tune_register,
        out_indices=frozen_indices,
        select_layers=tuned_indices,
        channels=vit_hidden_dims,
        tuning_type='frozen'),
    decode_head=dict(
        type='DconvUpsamplingBNHead',
        in_channels=vit_hidden_dims,
        channels=vit_hidden_dims,
        dropout_ratio=0,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)))
