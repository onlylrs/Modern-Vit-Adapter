# Source: ViT-Split, detection/configs/mask_rcnn/vitsplit/mask_rcnn_dinov2_vitsplit_base_fpn_1x_coco.py
# Adapted for this repository: frozen DINOv2 layers use explicit uniform
# sampling, and the split head keeps the official long tuned layer span.

_base_ = [
    '../../_base_/models/mask_rcnn_r50_fpn.py',
    '../../_base_/datasets/coco_instance.py',
    '../../_base_/schedules/schedule_1x.py',
    '../../_base_/default_runtime.py'
]

initial_lr = 1e-4
wd = 5e-2
layer_decay_rate = 0.7
vit_hidden_dims = 768
pergpu_batch_size = 1
total_layers = 12
frozen_count = 5


def uniform_indices(num_layers, count, start=0):
    if count == 1:
        return [num_layers - 1]
    gap = (num_layers - 1 - start) / float(count - 1)
    return [int(round(start + idx * gap)) for idx in range(count)]


frozen_indices = uniform_indices(total_layers, frozen_count, start=2)
tuned_indices = list(range(1, total_layers))
register_version = False
tune_register = False
find_unused_parameters = True

model = dict(
    backbone=dict(
        _delete_=True,
        type='DINOViTSplitFusion',
        backbone_size='base',
        register_version=register_version,
        tune_register=tune_register,
        out_indices=frozen_indices,
        select_layers=tuned_indices,
        channels=vit_hidden_dims,
        tuning_type='frozen'),
    neck=dict(
        type='FPN',
        in_channels=[vit_hidden_dims, vit_hidden_dims, vit_hidden_dims, vit_hidden_dims],
        out_channels=256,
        num_outs=5))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                                 (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                                 (736, 1333), (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True)
             ],
             [
                 dict(type='Resize',
                      img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(384, 600),
                      allow_negative_crop=True),
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                 (576, 1333), (608, 1333), (640, 1333),
                                 (672, 1333), (704, 1333), (736, 1333),
                                 (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      override=True,
                      keep_ratio=True)
             ]
         ]),
    dict(type='RandomCrop',
         crop_type='absolute_range',
         crop_size=(1024, 1024),
         allow_negative_crop=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
data = dict(samples_per_gpu=pergpu_batch_size, train=dict(pipeline=train_pipeline))
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=initial_lr,
    weight_decay=wd,
    constructor='LayerDecayOptimizerConstructorSplit',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=layer_decay_rate))
optimizer_config = dict(grad_clip=None)
log_config = dict(interval=50, hooks=[dict(type='PrintLrGroupHook', by_epoch=True)])
fp16 = dict(loss_scale=dict(init_scale=512))
checkpoint_config = dict(interval=1, max_keep_ckpts=3, save_last=True)
