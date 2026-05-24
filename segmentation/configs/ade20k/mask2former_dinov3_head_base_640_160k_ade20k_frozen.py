_base_ = './kaa_mask2former_dinov3_base_640_160k_ade20k_frozen_base.py'

model = dict(
    backbone=dict(
        _delete_=True,
        type='DINOv3SplitFusionSeg',
        pretrained='/home/rliuar/0_Storage/hf_home/hub/dinov3-vitb16-pretrain-lvd1689m',
        out_indices='uniform',
        out_indices_count=4,
        out_indices_start=2,
        select_layers=[9, 10, 11],
        channels=768,
        tuning_type='frozen'))

work_dir = 'work_dirs/ade20k/mask2former_dinov3_head_base_640_160k_ade20k_frozen'
