import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_automation import (
    bootstrap_ci_from_image_values,
    bootstrap_ci_from_coco_predictions,
    metric_with_ci,
)


def test_bootstrap_ci_from_image_values_is_deterministic():
    values = [0.1, 0.2, 0.4, 0.5, 0.9]

    ci_a = bootstrap_ci_from_image_values(values, n_resamples=200, seed=11)
    ci_b = bootstrap_ci_from_image_values(values, n_resamples=200, seed=11)
    ci_c = bootstrap_ci_from_image_values(values, n_resamples=200, seed=12)

    assert ci_a == ci_b
    assert not (
        math.isclose(ci_a["lower"], ci_c["lower"], rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(ci_a["upper"], ci_c["upper"], rel_tol=0.0, abs_tol=1e-12)
    )


def test_metric_with_ci_returns_machine_readable_shape():
    payload = metric_with_ci(point_estimate=0.75, ci_lower=0.6, ci_upper=0.85)
    assert payload == {
        "point_estimate": 0.75,
        "ci95": {"lower": 0.6, "upper": 0.85},
    }


def _toy_coco_detection_dataset():
    # Three images with one object each; predictions have varied IoU quality.
    gt = {
        "info": {"description": "toy"},
        "licenses": [],
        "images": [
            {"id": 1, "width": 20, "height": 20},
            {"id": 2, "width": 20, "height": 20},
            {"id": 3, "width": 20, "height": 20},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [2, 2, 10, 10], "area": 100, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [2, 2, 10, 10], "area": 100, "iscrowd": 0},
            {"id": 3, "image_id": 3, "category_id": 1, "bbox": [2, 2, 10, 10], "area": 100, "iscrowd": 0},
        ],
        "categories": [{"id": 1, "name": "cell"}],
    }
    preds = [
        # High-IoU match
        {"image_id": 1, "category_id": 1, "bbox": [2, 2, 10, 10], "score": 0.99},
        # Borderline for AP30 but not AP50 (IoU ~= 0.47)
        {"image_id": 2, "category_id": 1, "bbox": [4, 4, 10, 10], "score": 0.95},
        # Miss
        {"image_id": 3, "category_id": 1, "bbox": [14, 14, 4, 4], "score": 0.90},
    ]
    return gt, preds


def test_detection_coco_bootstrap_produces_coherent_point_and_ci():
    gt, preds = _toy_coco_detection_dataset()
    results = bootstrap_ci_from_coco_predictions(
        gt_dataset=gt,
        predictions=preds,
        iou_type="bbox",
        n_resamples=200,
        seed=7,
        include_ap30=True,
        include_mar=True,
    )

    for metric_name in ["mAP", "AP30", "AP50", "mAR"]:
        point = results[metric_name]["point_estimate"]
        lower = results[metric_name]["ci95"]["lower"]
        upper = results[metric_name]["ci95"]["upper"]
        assert lower <= point <= upper


def test_detection_coco_bootstrap_separates_ap30_and_ap50_intervals():
    gt, preds = _toy_coco_detection_dataset()
    results = bootstrap_ci_from_coco_predictions(
        gt_dataset=gt,
        predictions=preds,
        iou_type="bbox",
        n_resamples=200,
        seed=9,
        include_ap30=True,
        include_mar=True,
    )

    # Regression target: AP30 and AP50 should not collapse to identical CI bounds.
    ap30_ci = results["AP30"]["ci95"]
    ap50_ci = results["AP50"]["ci95"]
    assert not (
        np.isclose(ap30_ci["lower"], ap50_ci["lower"]) and np.isclose(ap30_ci["upper"], ap50_ci["upper"])
    )
    assert results["AP30"]["point_estimate"] >= results["AP50"]["point_estimate"]


def test_detection_coco_bootstrap_parallel_matches_serial():
    gt, preds = _toy_coco_detection_dataset()

    serial = bootstrap_ci_from_coco_predictions(
        gt_dataset=gt,
        predictions=preds,
        iou_type="bbox",
        n_resamples=40,
        seed=17,
        include_ap30=True,
        include_mar=True,
        n_jobs=1,
    )
    parallel = bootstrap_ci_from_coco_predictions(
        gt_dataset=gt,
        predictions=preds,
        iou_type="bbox",
        n_resamples=40,
        seed=17,
        include_ap30=True,
        include_mar=True,
        n_jobs=2,
    )

    assert parallel == serial


def _matching_edge_case_dataset(iou_type):
    from pycocotools import mask as mask_utils

    gt = {'images': [], 'annotations': [], 'categories': [
        {'id': 1, 'name': 'cell'}, {'id': 2, 'name': 'other'}], 'info': {}}
    predictions = []
    ann_id = 1
    for idx, image_id in enumerate([41, 8, 3, 90, 12]):
        gt['images'].append({'id': image_id, 'width': 32, 'height': 32})
        # Last image has no GT; category 2 is absent in some draws.
        if idx != 4:
            for category in ([1, 2] if idx == 0 else [1]):
                ann = {'id': ann_id, 'image_id': image_id, 'category_id': category,
                       'bbox': [2, 2, 10, 10], 'area': 100, 'iscrowd': int(idx == 2)}
                mask = np.zeros((32, 32), dtype=np.uint8)
                mask[2:12, 2:12] = 1
                ann['segmentation'] = mask_utils.encode(np.asfortranarray(mask))
                gt['annotations'].append(ann)
                ann_id += 1
        # Equal scores, false positives, and enough predictions to exercise maxDets.
        for n in range(105 if idx == 0 else 3):
            x = 2 if n == 0 else 5 + n % 12
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[2:12, x:x + 10] = 1
            pred = {'image_id': image_id, 'category_id': 1 if n % 2 == 0 else 2,
                    'score': 0.8 if n < 2 else 0.4, 'bbox': [x, 2, 10, 10]}
            if iou_type == 'segm':
                pred['segmentation'] = mask_utils.encode(np.asfortranarray(mask))
            predictions.append(pred)
    return gt, predictions


def test_cached_coco_matches_full_resampling_with_duplicates_crowds_and_ties():
    from eval_automation import (
        _build_coco_match_cache, _accumulate_cached_sample,
        _evaluate_coco_bootstrap_sample,
    )
    for kind in ('bbox', 'segm'):
        gt, predictions = _matching_edge_case_dataset(kind)
        for preds in (predictions, []):
            cache = _build_coco_match_cache(gt, preds, kind, True)
            for draw in ([8, 41, 8, 3, 12], [12] * 5, [3, 90, 3, 90, 8], [41] * 5):
                expected = _evaluate_coco_bootstrap_sample((gt, preds, draw, kind, True, True))
                actual = _accumulate_cached_sample(cache, draw, True)
                assert actual.keys() == expected.keys()
                for name in expected:
                    np.testing.assert_allclose(actual[name], expected[name], atol=1e-12, rtol=0)


def _dense_aji_reference(gt, pred):
    gt = [m.astype(bool) for m in gt if m.any()]
    pred = [m.astype(bool) for m in pred if m.any()]
    if not gt and not pred:
        return {'aji': 1.0, 'dice': 1.0}
    if not gt or not pred:
        return {'aji': 0.0, 'dice': 0.0}
    unmatched = set(range(len(pred)))
    intersection_total = union_total = 0
    for mask in gt:
        best_idx, best_iou = None, 0
        best_intersection = best_union = 0
        for idx in unmatched:
            intersection = np.logical_and(mask, pred[idx]).sum()
            union = np.logical_or(mask, pred[idx]).sum()
            iou = intersection / union
            if iou > best_iou:
                best_idx, best_iou = idx, iou
                best_intersection, best_union = intersection, union
        if best_idx is None:
            union_total += mask.sum()
        else:
            unmatched.remove(best_idx)
            intersection_total += best_intersection
            union_total += best_union
    union_total += sum(pred[idx].sum() for idx in unmatched)
    merged_gt, merged_pred = np.logical_or.reduce(gt), np.logical_or.reduce(pred)
    return {'aji': intersection_total / union_total,
            'dice': 2 * np.logical_and(merged_gt, merged_pred).sum() /
                    (merged_gt.sum() + merged_pred.sum())}


def test_rle_aji_preserves_dense_greedy_matching():
    from eval_automation import compute_aji_dice
    rng = np.random.default_rng(123)
    cases = [([], []), ([np.ones((24, 24), dtype=bool)], []),
             ([], [np.zeros((24, 24), dtype=bool)])]
    for _ in range(30):
        cases.append(([rng.random((24, 24)) > 0.8 for _ in range(5)],
                      [rng.random((24, 24)) > 0.8 for _ in range(7)]))
    # Identical IoUs exercise the existing tie-breaking behavior.
    mask = np.zeros((24, 24), dtype=bool)
    mask[2:12, 2:12] = True
    cases.append(([mask, mask], [mask, mask, mask]))
    for gt, pred in cases:
        expected, actual = _dense_aji_reference(gt, pred), compute_aji_dice(gt, pred)
        for name in expected:
            np.testing.assert_allclose(actual[name], expected[name], atol=1e-12, rtol=0)


def test_segmentation_predictions_without_bbox_are_not_mutated():
    import copy
    gt, preds = _matching_edge_case_dataset('segm')
    for pred in preds:
        del pred['bbox']
    original = copy.deepcopy(preds)
    first = bootstrap_ci_from_coco_predictions(gt, preds, 'segm', n_resamples=10)
    second = bootstrap_ci_from_coco_predictions(gt, preds, 'segm', n_resamples=10)
    assert first == second
    assert preds == original



def test_detection_mar_requires_correct_category():
    from eval_automation import _evaluate_coco_metric_set
    gt = {
        'info': {},
        'images': [{'id': 1, 'width': 32, 'height': 32}],
        'categories': [{'id': 1, 'name': 'WBC'}, {'id': 2, 'name': 'RBC'}],
        'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1,
                         'bbox': [2, 2, 10, 10], 'area': 100, 'iscrowd': 0}],
    }
    wrong = [{'image_id': 1, 'category_id': 2, 'bbox': [2, 2, 10, 10], 'score': 0.99}]
    right = [dict(wrong[0], category_id=1)]
    for predictions, expected in ((wrong, 0.0), (right, 1.0)):
        ordinary = _evaluate_coco_metric_set(gt, predictions, 'bbox', include_mar=True)
        bootstrapped = bootstrap_ci_from_coco_predictions(
            gt, predictions, 'bbox', n_resamples=10, include_mar=True)
        assert ordinary['mAR'] == expected
        assert bootstrapped['mAR']['point_estimate'] == expected
        assert bootstrapped['mAR']['original_point_estimate'] == expected
        assert bootstrapped['mAR']['ci95'] == {'lower': expected, 'upper': expected}
