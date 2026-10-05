"""Check AJI-only validation against full COCO validation on real result formatting."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pycocotools import mask as mask_utils
from mmdet.datasets import CocoDataset

MODULE_PATH = Path(__file__).resolve().parents[1] / 'mmdet_custom/datasets/crown_coco.py'
spec = importlib.util.spec_from_file_location('crown_validation_dataset', MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_aji_only_skips_coco_ap_and_preserves_scores(tmp_path):
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[3:13, 3:13] = 1
    ann = tmp_path / 'ann.json'
    ann.write_text(json.dumps({
        'info': {}, 'images': [{'id': 1, 'file_name': 'unused.png', 'width': 24, 'height': 24}],
        'categories': [{'id': 1, 'name': 'cell'}],
        'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1, 'iscrowd': 0,
                         'bbox': [3, 3, 10, 10], 'area': 100,
                         'segmentation': [[3, 3, 13, 3, 13, 13, 3, 13]]}],
    }))
    dataset = module.CrownInstanceCocoDataset(str(ann), pipeline=[], classes=('cell',), test_mode=True)
    results = [([np.array([[3, 3, 13, 13, 0.9]])],
                [[mask_utils.encode(np.asfortranarray(mask))]])]
    full = dataset.evaluate(results, metric='segm')
    with patch.object(CocoDataset, 'evaluate', side_effect=AssertionError('AP should be skipped')):
        selected = dataset.evaluate(results, metric='AJI')
    assert selected == {'AJI': full['AJI'], 'Dice': full['Dice']}
    assert selected['AJI'] == selected['Dice'] == 1.0
