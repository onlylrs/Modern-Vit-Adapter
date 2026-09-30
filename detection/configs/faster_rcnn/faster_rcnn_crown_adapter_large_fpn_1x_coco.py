"""CROWN ViT-L/16 Faster R-CNN template; replace dataset settings for each cohort."""

_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

pretrained = '/homes/rliuar/work/mnt/nas6/Cytology/Private/temp/X_Ckpts/crown_ckpts/CROWN.pth'
official_root = '/homes/rliuar/work/0_Official/CROWN'

model = dict(
    backbone=dict(
        _delete_=True, type='ViTAdapterCROWN', pretrained=pretrained,
        official_root=official_root, freeze_backbone=False,
        interaction_indexes=((0, 5), (6, 11), (12, 17), (18, 23)),
        deform_num_heads=16, deform_ratio=0.5, with_cp=True,
    ),
    neck=dict(type='FPN', in_channels=[1024] * 4, out_channels=256, num_outs=5),
)

# Matches CROWN's data/transforms.py evaluation and pretraining normalization.
# MMDetection reads BGR images; to_rgb=True converts them before normalization.
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1024, 640), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug', img_scale=(1024, 640), flip=False, transforms=[
        dict(type='Resize', keep_ratio=True),
        dict(type='RandomFlip'),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='Collect', keys=['img']),
    ]),
]
data = dict(samples_per_gpu=1, train=dict(pipeline=train_pipeline), val=dict(pipeline=test_pipeline),
            test=dict(pipeline=test_pipeline))
optimizer = dict(_delete_=True, type='AdamW', lr=1e-4, weight_decay=0.05)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=1.0, norm_type=2))
