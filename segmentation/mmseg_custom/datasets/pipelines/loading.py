import numpy as np
from mmseg.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadCocoInstanceAnnotations(object):
    """Load semantic and instance maps from ``ann_info``."""

    def __call__(self, results):
        from ..coco_instance import CocoInstanceDatasetBridge

        ann_info = results['ann_info']
        dense_targets = CocoInstanceDatasetBridge.build_dense_targets(ann_info)
        gt_semantic_seg = dense_targets['gt_semantic_seg']
        gt_instance_map = dense_targets['gt_instance_map']
        results['gt_semantic_seg'] = gt_semantic_seg.astype(np.int64)
        results['gt_instance_map'] = gt_instance_map.astype(np.int64)
        results.setdefault('seg_fields', [])
        results['seg_fields'].append('gt_semantic_seg')
        results['seg_fields'].append('gt_instance_map')
        return results

    def __repr__(self):
        return self.__class__.__name__
