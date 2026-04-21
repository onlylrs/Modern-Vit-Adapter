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
