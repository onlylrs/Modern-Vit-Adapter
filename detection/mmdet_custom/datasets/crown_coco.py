"""COCO instance dataset that exposes AJI to the validation hook."""

from pathlib import Path
from tempfile import TemporaryDirectory

from mmdet.datasets import CocoDataset
from mmdet.datasets.builder import DATASETS


@DATASETS.register_module()
class CrownInstanceCocoDataset(CocoDataset):
    def evaluate(self, results, metric='segm', **kwargs):
        metrics = super().evaluate(results, metric=metric, **kwargs)
        if metric != 'segm':
            return metrics
        from eval_automation import compute_aji_dice_from_coco_json

        with TemporaryDirectory(prefix='crown_val_') as temporary:
            result_files, _ = self.format_results(
                results, jsonfile_prefix=str(Path(temporary) / 'results'))
            if 'segm' not in result_files:
                raise ValueError('Mask R-CNN results must contain instance masks for AJI')
            scores = compute_aji_dice_from_coco_json(self.ann_file, result_files['segm'])
        metrics['AJI'] = scores['aji']
        metrics['Dice'] = scores['dice']
        return metrics
