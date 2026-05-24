_base_ = './kaa_mask2former_dinov3_base_640_160k_ade20k_frozen_base.py'

model = dict(
    backbone=dict(
        kaa_variant='product',
        align_loss_weight=0.05,
        align_sample_size=1024))

work_dir = 'work_dirs/ade20k/kaa_mask2former_dinov3_base_640_160k_ade20k_frozen_align'
