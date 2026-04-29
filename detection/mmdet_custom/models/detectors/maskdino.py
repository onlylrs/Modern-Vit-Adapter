from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
from mmdet.models.detectors.base import BaseDetector


@DETECTORS.register_module()
class MaskDINO(BaseDetector):
    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if pretrained is not None:
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck) if neck is not None else None
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        x = self.extract_feat(img)
        return self.bbox_head(x)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks=None,
                      gt_bboxes_ignore=None,
                      gt_semantic_seg=None,
                      **kwargs):
        super().forward_train(img, img_metas)
        x = self.extract_feat(img)
        return self.bbox_head.forward_train(
            x, img_metas, gt_bboxes, gt_labels, gt_masks, gt_bboxes_ignore,
            gt_semantic_seg=gt_semantic_seg)

    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)
        return self.bbox_head.simple_test(x, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError('MaskDINO does not support aug_test yet.')

    def onnx_export(self, img, img_metas):
        raise NotImplementedError
