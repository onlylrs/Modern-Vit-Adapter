_base_ = [
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py',
]

norm_cfg = dict(type='BN', requires_grad=False)
model = dict(
    type='EncoderDecoderMask2Former',
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=norm_cfg,
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/path/to/resnet50-0676ba61.pth')),
    decode_head=dict(
        type='MaskDINOSemanticHead',
        in_channels=[256, 512, 1024, 2048],
        in_index=[0, 1, 2, 3],
        strides=[4, 8, 16, 32],
        feat_channels=256,
        mask_dim=256,
        num_classes=150,
        num_queries=100,
        pixel_decoder=dict(
            transformer_in_indices=(1, 2, 3),
            transformer_dropout=0.0,
            transformer_nheads=8,
            transformer_dim_feedforward=1024,
            transformer_enc_layers=6,
            common_stride=4,
            num_feature_levels=3,
            total_num_feature_levels=3,
            feature_order='high2low'),
        transformer_decoder=dict(
            nheads=8,
            dim_feedforward=2048,
            dec_layers=9,
            enforce_input_project=False,
            two_stage=False,
            dn='seg',
            noise_scale=0.4,
            dn_num=100,
            initialize_box_type='no',
            initial_pred=True,
            learn_tgt=False,
            total_num_feature_levels=3,
            dropout=0.0,
            dec_n_points=4,
            semantic_ce_loss=True),
        loss_cfg=dict(
            losses=['labels', 'masks'],
            dn_losses=['labels', 'masks'],
            class_weight=4.0,
            mask_weight=5.0,
            dice_weight=5.0,
            cost_class=4.0,
            cost_mask=5.0,
            cost_dice=5.0,
            no_object_weight=0.1,
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75),
        align_corners=False),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

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
    step=[135000, 150000],
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=5000, max_keep_ckpts=1,
                         save_last=True)
evaluation = dict(interval=5000, metric='mIoU', pre_eval=True, save_best='mIoU')
fp16 = dict(loss_scale=dict(init_scale=512))
# load_from = '/path/to/maskdino_r50_50ep_ade20k_semantic_mmseg_converted.pth' See readme to download it.
work_dir = 'work_dirs/segmentation/ade20k_maskdino_r50_semantic'
