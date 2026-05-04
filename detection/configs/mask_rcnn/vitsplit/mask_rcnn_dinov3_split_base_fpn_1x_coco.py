# DINOv3 adaptation of the ViT-Split Mask R-CNN config.
# Inspired by ViT-Split, with DINOv3 checkpoint loading from this repository.

_base_ = ['./mask_rcnn_dinov2_vitsplit_base_fpn_1x_coco.py']

pretrained = '/home/rliuar/0_Storage/hf_home/hub/dinov3-vitb16-pretrain-lvd1689m'

model = dict(
    backbone=dict(
        _delete_=True,
        type='DINOv3SplitFusion',
        pretrained=pretrained,
        out_indices='uniform',
        out_indices_count=5,
        out_indices_start=2,
        select_layers=list(range(1, 12)),
        channels=768,
        tuning_type='frozen'),
    neck=dict(
        type='FPN',
        in_channels=[768, 768, 768, 768],
        out_channels=256,
        num_outs=5))
