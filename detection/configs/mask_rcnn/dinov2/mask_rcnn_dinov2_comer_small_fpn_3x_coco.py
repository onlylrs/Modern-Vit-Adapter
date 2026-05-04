_base_ = ['./mask_rcnn_dinov2_comer_small_fpn_1x_coco.py']

lr_config = dict(step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)
