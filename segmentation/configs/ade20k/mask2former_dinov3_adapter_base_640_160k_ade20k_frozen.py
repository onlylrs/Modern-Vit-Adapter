_base_ = './kaa_mask2former_dinov3_base_640_160k_ade20k_frozen_base.py'

model = dict(
    backbone=dict(
        _delete_=True,
        type='ViTAdapterDINOv3Seg',
        pretrain_size=640,
        img_size=640,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        drop_path_rate=0.0,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=12,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        pretrained='/home/rliuar/0_Storage/hf_home/hub/dinov3-vitb16-pretrain-lvd1689m',
        freeze_backbone=True,
        adapter_mode='vit_adapter'))

work_dir = 'work_dirs/ade20k/mask2former_dinov3_adapter_base_640_160k_ade20k_frozen'
