# DINOv3 adaptation of the ViT-Split linear ADE20K config.
# Inspired by ViT-Split, with DINOv3 checkpoint loading from this repository.

_base_ = ['./linear_dinov2_vitsplit_base_512_40k_ade20k.py']

pretrained = '/home/rliuar/0_Storage/hf_home/hub/dinov3-vitb16-pretrain-lvd1689m'

model = dict(
    backbone=dict(
        type='DINOv3SplitFusionSeg',
        pretrained=pretrained,
        out_indices='uniform',
        out_indices_count=4,
        out_indices_start=2,
        select_layers=[9, 10, 11],
        channels=768,
        tuning_type='frozen'))
