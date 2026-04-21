import argparse
import copy
import io
import contextlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils


def parse_epoch_from_checkpoint_name(name):
    match = re.search(r"epoch_(\d+)", name)
    if match is None:
        return None
    return int(match.group(1))


def collect_checkpoints(experiment_dir, select="all"):
    experiment_dir = Path(experiment_dir)
    candidates = list(experiment_dir.glob("*.pth"))
    if not candidates:
        return []

    epoch_ckpts = []
    best_ckpts = []
    latest_ckpts = []
    other_ckpts = []

    for ckpt in candidates:
        name = ckpt.name
        if re.match(r"^epoch_\d+\.pth$", name):
            epoch_ckpts.append(ckpt)
        elif name.startswith("best_"):
            best_ckpts.append(ckpt)
        elif name == "latest.pth":
            latest_ckpts.append(ckpt)
        else:
            other_ckpts.append(ckpt)

    epoch_ckpts.sort(key=lambda p: parse_epoch_from_checkpoint_name(p.name) or -1)
    best_ckpts.sort(key=lambda p: (parse_epoch_from_checkpoint_name(p.name) or -1, p.name))
    other_ckpts.sort(key=lambda p: p.name)

    ordered = epoch_ckpts + best_ckpts + other_ckpts + latest_ckpts
    if select == "all":
        return ordered
    if select == "latest":
        return latest_ckpts[-1:] if latest_ckpts else ordered[-1:]
    if select == "best":
        if best_ckpts:
            return best_ckpts
        return latest_ckpts[-1:] if latest_ckpts else ordered[-1:]
    if select == "final":
        if best_ckpts:
            return [best_ckpts[-1]]
        return latest_ckpts[-1:] if latest_ckpts else ordered[-1:]

    raise ValueError(f"Unsupported checkpoint selection mode: {select}")


def _merge_instance_masks(instance_masks):
    if not instance_masks:
        return None
    merged = np.zeros_like(instance_masks[0], dtype=bool)
    for m in instance_masks:
        merged |= m
    return merged


def compute_aji_dice(gt_instance_masks, pred_instance_masks):
    gt = [np.asarray(m, dtype=bool) for m in gt_instance_masks if np.any(m)]
    pred = [np.asarray(m, dtype=bool) for m in pred_instance_masks if np.any(m)]

    if not gt and not pred:
        return {"aji": 1.0, "dice": 1.0}
    if not gt or not pred:
        return {"aji": 0.0, "dice": 0.0}

    unmatched_pred = set(range(len(pred)))
    aji_intersection = 0.0
    aji_union = 0.0

    for gt_mask in gt:
        best_idx = None
        best_intersection = 0.0
        best_union = 0.0
        best_iou = 0.0
        for pred_idx in unmatched_pred:
            pred_mask = pred[pred_idx]
            intersection = float(np.logical_and(gt_mask, pred_mask).sum())
            union = float(np.logical_or(gt_mask, pred_mask).sum())
            if union <= 0.0:
                continue
            iou = intersection / union
            if iou > best_iou:
                best_iou = iou
                best_idx = pred_idx
                best_intersection = intersection
                best_union = union

        if best_idx is not None and best_iou > 0.0:
            unmatched_pred.remove(best_idx)
            aji_intersection += best_intersection
            aji_union += best_union
        else:
            aji_union += float(gt_mask.sum())

    for pred_idx in unmatched_pred:
        aji_union += float(pred[pred_idx].sum())

    aji = 0.0 if aji_union == 0.0 else aji_intersection / aji_union

    gt_union = _merge_instance_masks(gt)
    pred_union = _merge_instance_masks(pred)
    intersection = float(np.logical_and(gt_union, pred_union).sum())
    denominator = float(gt_union.sum() + pred_union.sum())
    dice = 0.0 if denominator == 0.0 else (2.0 * intersection / denominator)
    return {"aji": aji, "dice": dice}


def _decode_annotation_to_mask(annotation, height, width):
    segmentation = annotation.get("segmentation", None)
    if segmentation is None:
        return None
    if isinstance(segmentation, list):
        if len(segmentation) == 0:
            return None
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(segmentation, dict) and isinstance(segmentation.get("counts"), list):
        rle = mask_utils.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation
    return np.asarray(mask_utils.decode(rle), dtype=bool)


def _decode_prediction_to_mask(prediction, height, width):
    seg = prediction.get("segmentation", None)
    if seg is None:
        return None
    if isinstance(seg, list):
        if len(seg) == 0:
            return None
        rles = mask_utils.frPyObjects(seg, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(seg, dict) and isinstance(seg.get("counts"), list):
        rle = mask_utils.frPyObjects(seg, height, width)
    else:
        rle = seg
    return np.asarray(mask_utils.decode(rle), dtype=bool)


def compute_aji_dice_from_coco_json(ann_file, pred_file):
    with open(ann_file, "r", encoding="utf-8") as f:
        ann = json.load(f)
    with open(pred_file, "r", encoding="utf-8") as f:
        preds = json.load(f)

    values = compute_aji_dice_image_values_from_coco_dict(ann, preds)
    if not values["aji_values"]:
        return {"aji": 0.0, "dice": 0.0}
    return {"aji": float(np.mean(values["aji_values"])), "dice": float(np.mean(values["dice_values"]))}


def compute_aji_dice_image_values_from_coco_dict(gt_dataset, predictions):
    image_meta = {}
    for image in gt_dataset.get("images", []):
        image_meta[image["id"]] = (int(image["height"]), int(image["width"]))

    gt_by_image = {}
    for annotation in gt_dataset.get("annotations", []):
        if int(annotation.get("iscrowd", 0)) == 1:
            continue
        image_id = annotation["image_id"]
        if image_id not in image_meta:
            continue
        height, width = image_meta[image_id]
        decoded = _decode_annotation_to_mask(annotation, height, width)
        if decoded is None or not np.any(decoded):
            continue
        gt_by_image.setdefault(image_id, []).append(decoded)

    pred_by_image = {}
    for prediction in predictions:
        image_id = prediction["image_id"]
        if image_id not in image_meta:
            continue
        height, width = image_meta[image_id]
        decoded = _decode_prediction_to_mask(prediction, height, width)
        if decoded is None or not np.any(decoded):
            continue
        pred_by_image.setdefault(image_id, []).append(decoded)

    all_image_ids = sorted(set(image_meta.keys()))
    aji_values = []
    dice_values = []
    for image_id in all_image_ids:
        local = compute_aji_dice(gt_by_image.get(image_id, []), pred_by_image.get(image_id, []))
        aji_values.append(float(local["aji"]))
        dice_values.append(float(local["dice"]))
    return {"aji_values": aji_values, "dice_values": dice_values}


def _safe_float(metric_dict, key):
    value = metric_dict.get(key, None)
    if value is None:
        return None
    return float(value)


def bootstrap_ci_from_image_values(values, n_resamples=1000, seed=42):
    values_arr = np.asarray(values, dtype=np.float64)
    if values_arr.size == 0:
        return {"lower": None, "upper": None}
    if n_resamples <= 0:
        raise ValueError("n_resamples must be > 0")

    rng = np.random.default_rng(seed)
    n = values_arr.shape[0]
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        sample_idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(values_arr[sample_idx]))
    return {
        "lower": float(np.percentile(means, 2.5)),
        "upper": float(np.percentile(means, 97.5)),
    }


def metric_with_ci(point_estimate, ci_lower, ci_upper):
    return {
        "point_estimate": None if point_estimate is None else float(point_estimate),
        "ci95": {
            "lower": None if ci_lower is None else float(ci_lower),
            "upper": None if ci_upper is None else float(ci_upper),
        },
    }


def metric_with_bootstrap_values(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return metric_with_ci(None, None, None)
    return metric_with_ci(
        point_estimate=float(np.mean(arr)),
        ci_lower=float(np.percentile(arr, 2.5)),
        ci_upper=float(np.percentile(arr, 97.5)),
    )


def _build_coco_api_from_dataset_dict(gt_dataset):
    coco_gt = COCO()
    coco_gt.dataset = copy.deepcopy(gt_dataset)
    coco_gt.createIndex()
    return coco_gt


def _load_coco_results(coco_gt, predictions):
    if predictions:
        return coco_gt.loadRes(predictions)

    # pycocotools expects at least one prediction-like entry. Use impossible score to ensure no matches.
    categories = coco_gt.dataset.get("categories", [])
    category_id = int(categories[0]["id"]) if categories else 1
    sentinel = [{"image_id": int(coco_gt.getImgIds()[0]), "category_id": category_id, "bbox": [0, 0, 1, 1], "score": -1e9}]
    return coco_gt.loadRes(sentinel)


def _evaluate_coco_metric_set(gt_dataset, predictions, iou_type, include_ap30=False, include_mar=False):
    coco_gt = _build_coco_api_from_dataset_dict(gt_dataset)
    coco_dt = _load_coco_results(coco_gt, predictions)

    with contextlib.redirect_stdout(io.StringIO()):
        eval_default = COCOeval(coco_gt, coco_dt, iouType=iou_type)
        eval_default.params.imgIds = sorted(coco_gt.getImgIds())
        eval_default.evaluate()
        eval_default.accumulate()
        eval_default.summarize()

    out = {
        "mAP": float(eval_default.stats[0]),
        "AP50": float(eval_default.stats[1]),
    }

    if include_ap30:
        with contextlib.redirect_stdout(io.StringIO()):
            eval_ap30 = COCOeval(coco_gt, coco_dt, iouType=iou_type)
            eval_ap30.params.imgIds = sorted(coco_gt.getImgIds())
            eval_ap30.params.iouThrs = np.array([0.3], dtype=np.float64)
            eval_ap30.evaluate()
            eval_ap30.accumulate()
            eval_ap30.summarize()
        out["AP30"] = float(eval_ap30.stats[0])

    if include_mar:
        with contextlib.redirect_stdout(io.StringIO()):
            eval_mar = COCOeval(coco_gt, coco_dt, iouType=iou_type)
            eval_mar.params.imgIds = sorted(coco_gt.getImgIds())
            eval_mar.params.useCats = 0
            eval_mar.evaluate()
            eval_mar.accumulate()
            eval_mar.summarize()
        out["mAR"] = float(eval_mar.stats[8])
    return out


def _bootstrap_coco_resample(gt_dataset, predictions, sampled_image_ids):
    images_by_id = {int(img["id"]): img for img in gt_dataset.get("images", [])}
    anns_by_image = {}
    for ann in gt_dataset.get("annotations", []):
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)
    preds_by_image = {}
    for pred in predictions:
        preds_by_image.setdefault(int(pred["image_id"]), []).append(pred)

    remapped_images = []
    remapped_annotations = []
    remapped_predictions = []
    next_ann_id = 1
    for idx, old_image_id in enumerate(sampled_image_ids, start=1):
        base_img = copy.deepcopy(images_by_id[int(old_image_id)])
        base_img["id"] = idx
        remapped_images.append(base_img)

        for ann in anns_by_image.get(int(old_image_id), []):
            new_ann = copy.deepcopy(ann)
            new_ann["id"] = next_ann_id
            next_ann_id += 1
            new_ann["image_id"] = idx
            remapped_annotations.append(new_ann)

        for pred in preds_by_image.get(int(old_image_id), []):
            new_pred = copy.deepcopy(pred)
            new_pred["image_id"] = idx
            remapped_predictions.append(new_pred)

    remapped_gt = {
        "images": remapped_images,
        "annotations": remapped_annotations,
        "categories": copy.deepcopy(gt_dataset.get("categories", [])),
    }
    if "info" in gt_dataset:
        remapped_gt["info"] = copy.deepcopy(gt_dataset["info"])
    if "licenses" in gt_dataset:
        remapped_gt["licenses"] = copy.deepcopy(gt_dataset["licenses"])
    return remapped_gt, remapped_predictions


def bootstrap_ci_from_coco_predictions(
    gt_dataset,
    predictions,
    iou_type,
    n_resamples=1000,
    seed=42,
    include_ap30=False,
    include_mar=False,
    progress_label=None,
):
    image_ids = [int(img["id"]) for img in gt_dataset.get("images", [])]
    if not image_ids:
        metric_names = ["mAP", "AP50"]
        if include_ap30:
            metric_names.append("AP30")
        if include_mar:
            metric_names.append("mAR")
        return {name: metric_with_ci(None, None, None) for name in metric_names}
    if n_resamples <= 0:
        raise ValueError("n_resamples must be > 0")

    point_metrics = _evaluate_coco_metric_set(
        gt_dataset=gt_dataset,
        predictions=predictions,
        iou_type=iou_type,
        include_ap30=include_ap30,
        include_mar=include_mar,
    )
    metric_names = list(point_metrics.keys())
    bootstrap_samples = {name: [] for name in metric_names}

    rng = np.random.default_rng(seed)
    n_images = len(image_ids)
    progress_step = max(1, n_resamples // 10)
    for sample_idx in range(n_resamples):
        sampled_ids = [image_ids[int(i)] for i in rng.integers(0, n_images, size=n_images)]
        resampled_gt, resampled_preds = _bootstrap_coco_resample(gt_dataset, predictions, sampled_ids)
        sample_metrics = _evaluate_coco_metric_set(
            gt_dataset=resampled_gt,
            predictions=resampled_preds,
            iou_type=iou_type,
            include_ap30=include_ap30,
            include_mar=include_mar,
        )
        for name in metric_names:
            bootstrap_samples[name].append(float(sample_metrics[name]))
        if progress_label and (
            sample_idx == 0
            or sample_idx + 1 == n_resamples
            or (sample_idx + 1) % progress_step == 0
        ):
            print(
                f"[bootstrap] {progress_label}: {sample_idx + 1}/{n_resamples}",
                flush=True,
            )

    merged = {}
    for name in metric_names:
        values = np.asarray(bootstrap_samples[name], dtype=np.float64)
        merged[name] = metric_with_bootstrap_values(values)
        merged[name]["original_point_estimate"] = float(point_metrics[name])
    return merged


def extract_detection_metrics(metric_dict):
    return {
        "mAP": _safe_float(metric_dict, "bbox_mAP"),
        "AP30": _safe_float(metric_dict, "bbox_mAP_30"),
        "AP50": _safe_float(metric_dict, "bbox_mAP_50"),
        "mAR": _safe_float(metric_dict, "AR@100"),
    }


def extract_segm_metrics(metric_dict):
    return {
        "mAP": _safe_float(metric_dict, "segm_mAP"),
        "AP50": _safe_float(metric_dict, "segm_mAP_50"),
    }


def build_ap30_eval_options(extra_eval_options):
    base = ["iou_thrs=[0.3]"]
    if extra_eval_options:
        base.extend(extra_eval_options)
    return base


def build_mar_eval_options(extra_eval_options):
    base = ["proposal_nums=[100,300,1000]"]
    if extra_eval_options:
        base.extend(extra_eval_options)
    return base


def build_runtime_cfg_options(low_mem=False, samples_per_gpu=1, workers_per_gpu=1):
    if not low_mem:
        return []
    return [
        f"data.test.samples_per_gpu={int(samples_per_gpu)}",
        f"data.workers_per_gpu={int(workers_per_gpu)}",
    ]


def build_runtime_env(cuda_visible_devices=None, low_mem=False, append_existing_pythonpath=True):
    env = os.environ.copy()
    repo_root = str(Path(__file__).resolve().parents[1])
    if append_existing_pythonpath:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{repo_root}:{existing}" if existing else repo_root
    else:
        env["PYTHONPATH"] = repo_root

    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    if low_mem:
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return env


def run_detection_test(
    config_path,
    checkpoint_path,
    work_dir,
    metrics,
    eval_options,
    cfg_options=None,
    low_mem=False,
    cuda_visible_devices=None,
    python_executable=None,
):
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    out_file = work_dir / f"raw_outputs_{timestamp}.pkl"

    command = [
        str(python_executable) if python_executable else sys.executable,
        "detection/test.py",
        str(config_path),
        str(checkpoint_path),
        "--work-dir",
        str(work_dir),
        "--out",
        str(out_file),
        "--eval",
    ]
    command.extend(metrics)
    if eval_options:
        command.append("--eval-options")
        command.extend(eval_options)
    if cfg_options:
        command.append("--cfg-options")
        command.extend(cfg_options)

    env = build_runtime_env(cuda_visible_devices=cuda_visible_devices, low_mem=low_mem)

    subprocess.run(command, check=True, env=env)

    eval_json_files = sorted(work_dir.glob("eval_*.json"), key=lambda p: p.stat().st_mtime)
    if not eval_json_files:
        raise RuntimeError(f"No eval_*.json found in {work_dir}")
    return out_file, eval_json_files[-1]


def run_detection_format_only(
    config_path,
    checkpoint_path,
    work_dir,
    jsonfile_prefix,
    extra_eval_options=None,
    cfg_options=None,
    low_mem=False,
    cuda_visible_devices=None,
    python_executable=None,
):
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    command = [
        str(python_executable) if python_executable else sys.executable,
        "detection/test.py",
        str(config_path),
        str(checkpoint_path),
        "--work-dir",
        str(work_dir),
        "--format-only",
        "--eval-options",
        f"jsonfile_prefix={jsonfile_prefix}",
    ]
    if extra_eval_options:
        command.extend(extra_eval_options)
    if cfg_options:
        command.append("--cfg-options")
        command.extend(cfg_options)

    env = build_runtime_env(cuda_visible_devices=cuda_visible_devices, low_mem=low_mem)
    subprocess.run(command, check=True, env=env)


def _dataset_to_coco_gt_dict(dataset):
    if hasattr(dataset, "coco") and hasattr(dataset.coco, "dataset"):
        return copy.deepcopy(dataset.coco.dataset)

    ann_file = getattr(dataset, "ann_file", None)
    if ann_file is None:
        raise RuntimeError("Unable to resolve COCO ground truth: dataset has neither .coco.dataset nor .ann_file")
    with open(ann_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_coco_result_json(jsonfile_prefix, iou_type):
    prefix = Path(jsonfile_prefix)
    candidates = [Path(str(prefix) + f".{iou_type}.json"), Path(str(prefix) + ".json")]
    for path in candidates:
        if path.exists():
            return path
    raise RuntimeError(f"Could not find COCO result JSON for iou_type={iou_type} at prefix {jsonfile_prefix}")


def _build_eval_option_pairs(options):
    if not options:
        return []
    pairs = []
    for item in options:
        if "=" not in item:
            raise ValueError(f"Invalid eval-option '{item}', expected key=value")
        pairs.append(item)
    return pairs


def _build_test_dataset(config_path):
    from mmcv import Config
    from mmdet.datasets import build_dataset

    cfg = Config.fromfile(str(config_path))
    test_cfg = cfg.data.test
    if isinstance(test_cfg, dict):
        test_cfg = test_cfg.copy()
        test_cfg.test_mode = True
    else:
        test_cfg = [x.copy() for x in test_cfg]
        for ds_cfg in test_cfg:
            ds_cfg.test_mode = True
    dataset = build_dataset(test_cfg)
    return cfg, dataset


def _prediction_masks_from_output(output):
    if isinstance(output, tuple):
        if len(output) < 2:
            return []
        segm_result = output[1]
    else:
        segm_result = output
    if isinstance(segm_result, tuple):
        segm_result = segm_result[0]

    instance_masks = []
    for class_masks in segm_result:
        for encoded_mask in class_masks:
            if isinstance(encoded_mask, np.ndarray):
                mask = np.asarray(encoded_mask, dtype=bool)
            else:
                mask = np.asarray(mask_utils.decode(encoded_mask), dtype=bool)
            if np.any(mask):
                instance_masks.append(mask)
    return instance_masks


def _bbox_iou_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in box_b[:4]]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter_area
    if denom <= 0.0:
        return 0.0
    return inter_area / denom


def _match_precision_recall(gt_instances, pred_instances, iou_threshold, max_predictions=None):
    if max_predictions is not None:
        pred_instances = sorted(pred_instances, key=lambda x: x[2], reverse=True)[: int(max_predictions)]
    else:
        pred_instances = sorted(pred_instances, key=lambda x: x[2], reverse=True)

    if not gt_instances and not pred_instances:
        return 1.0, 1.0

    matched = set()
    tp = 0
    fp = 0
    for pred_cls, pred_obj, _score in pred_instances:
        best_idx = None
        best_iou = 0.0
        for idx, (gt_cls, gt_obj) in enumerate(gt_instances):
            if idx in matched or gt_cls != pred_cls:
                continue
            if isinstance(pred_obj, np.ndarray):
                iou = float(np.logical_and(pred_obj, gt_obj).sum()) / max(
                    float(np.logical_or(pred_obj, gt_obj).sum()), 1.0
                )
            else:
                iou = _bbox_iou_xyxy(pred_obj, gt_obj)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_idx is not None and best_iou >= iou_threshold:
            matched.add(best_idx)
            tp += 1
        else:
            fp += 1

    fn = max(len(gt_instances) - len(matched), 0)
    precision = 0.0 if tp + fp == 0 else tp / float(tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / float(tp + fn)
    return precision, recall


def _prediction_bboxes_from_output(output):
    if isinstance(output, tuple):
        bbox_result = output[0]
    else:
        bbox_result = output

    preds = []
    for cls_idx, class_boxes in enumerate(bbox_result):
        class_boxes = np.asarray(class_boxes)
        for row in class_boxes:
            if row.shape[0] < 4:
                continue
            score = float(row[4]) if row.shape[0] >= 5 else 1.0
            preds.append((int(cls_idx), np.asarray(row[:4], dtype=np.float64), score))
    return preds


def _prediction_masks_and_scores_from_output(output):
    bbox_result = output[0] if isinstance(output, tuple) else None
    segm_result = output[1] if isinstance(output, tuple) and len(output) > 1 else output
    if isinstance(segm_result, tuple):
        segm_result = segm_result[0]

    preds = []
    for cls_idx, class_masks in enumerate(segm_result):
        class_boxes = None
        if bbox_result is not None and cls_idx < len(bbox_result):
            class_boxes = np.asarray(bbox_result[cls_idx])
        for mask_idx, encoded_mask in enumerate(class_masks):
            if isinstance(encoded_mask, np.ndarray):
                mask = np.asarray(encoded_mask, dtype=bool)
            else:
                mask = np.asarray(mask_utils.decode(encoded_mask), dtype=bool)
            if not np.any(mask):
                continue
            score = 1.0
            if class_boxes is not None and mask_idx < class_boxes.shape[0] and class_boxes.shape[1] >= 5:
                score = float(class_boxes[mask_idx][4])
            preds.append((int(cls_idx), mask, score))
    return preds


def _compute_detection_image_metrics(dataset, outputs):
    metric_values = {"mAP": [], "AP30": [], "AP50": [], "mAR": []}
    iou_thresholds = np.arange(0.5, 1.0, 0.05)

    for idx, output in enumerate(outputs):
        ann_info = dataset.get_ann_info(idx)
        gt_bboxes = np.asarray(ann_info.get("bboxes", []))
        gt_labels = np.asarray(ann_info.get("labels", []), dtype=np.int64)
        gt_instances = [(int(gt_labels[i]), np.asarray(gt_bboxes[i], dtype=np.float64)) for i in range(len(gt_labels))]

        pred_instances = _prediction_bboxes_from_output(output)
        precisions = []
        for thr in iou_thresholds:
            precision, _ = _match_precision_recall(gt_instances, pred_instances, iou_threshold=float(thr))
            precisions.append(precision)

        p30, _ = _match_precision_recall(gt_instances, pred_instances, iou_threshold=0.3)
        p50, r50 = _match_precision_recall(gt_instances, pred_instances, iou_threshold=0.5, max_predictions=100)

        metric_values["mAP"].append(float(np.mean(precisions)))
        metric_values["AP30"].append(float(p30))
        metric_values["AP50"].append(float(p50))
        metric_values["mAR"].append(float(r50))

    return metric_values


def _compute_segmentation_image_metrics(dataset, outputs):
    metric_values = {"mAP": [], "AP50": [], "AJI": [], "Dice": []}
    iou_thresholds = np.arange(0.5, 1.0, 0.05)

    for idx, output in enumerate(outputs):
        ann_info = dataset.get_ann_info(idx)
        image_info = dataset.data_infos[idx]
        height = int(image_info["height"])
        width = int(image_info["width"])

        gt_labels = np.asarray(ann_info.get("labels", []), dtype=np.int64)
        gt_instances = []
        gt_instance_masks = []
        for mask_idx, raw_mask in enumerate(ann_info.get("masks", [])):
            decoded = _decode_annotation_to_mask({"segmentation": raw_mask}, height, width)
            if decoded is None or not np.any(decoded):
                continue
            cls_idx = int(gt_labels[mask_idx]) if mask_idx < len(gt_labels) else 0
            gt_instances.append((cls_idx, decoded))
            gt_instance_masks.append(decoded)

        pred_instances = _prediction_masks_and_scores_from_output(output)
        pred_instance_masks = [mask for _, mask, _ in pred_instances]

        precisions = []
        for thr in iou_thresholds:
            precision, _ = _match_precision_recall(gt_instances, pred_instances, iou_threshold=float(thr))
            precisions.append(precision)
        p50, _ = _match_precision_recall(gt_instances, pred_instances, iou_threshold=0.5)
        aji_dice = compute_aji_dice(gt_instance_masks, pred_instance_masks)

        metric_values["mAP"].append(float(np.mean(precisions)))
        metric_values["AP50"].append(float(p50))
        metric_values["AJI"].append(float(aji_dice["aji"]))
        metric_values["Dice"].append(float(aji_dice["dice"]))

    return metric_values


def _load_dataset_and_outputs(config_path, output_file):
    import mmcv

    _, dataset = _build_test_dataset(config_path)
    outputs = mmcv.load(str(output_file))
    if len(outputs) != len(dataset):
        raise RuntimeError(
            f"Output count ({len(outputs)}) does not match dataset size ({len(dataset)})"
        )
    return dataset, outputs


def compute_aji_dice_from_outputs(config_path, output_file):
    import mmcv

    _, dataset = _build_test_dataset(config_path)
    outputs = mmcv.load(str(output_file))

    if len(outputs) != len(dataset):
        raise RuntimeError(
            f"Output count ({len(outputs)}) does not match dataset size ({len(dataset)})"
        )

    aji_values = []
    dice_values = []
    for idx, output in enumerate(outputs):
        ann_info = dataset.get_ann_info(idx)
        image_info = dataset.data_infos[idx]
        height = int(image_info["height"])
        width = int(image_info["width"])
        gt_instance_masks = []
        for raw_mask in ann_info.get("masks", []):
            decoded = _decode_annotation_to_mask(
                {"segmentation": raw_mask},
                height,
                width,
            )
            if decoded is not None and np.any(decoded):
                gt_instance_masks.append(decoded)

        pred_instance_masks = _prediction_masks_from_output(output)
        local = compute_aji_dice(gt_instance_masks, pred_instance_masks)
        aji_values.append(local["aji"])
        dice_values.append(local["dice"])

    if not aji_values:
        return {"aji": 0.0, "dice": 0.0}
    return {"aji": float(np.mean(aji_values)), "dice": float(np.mean(dice_values))}


def compute_aji_dice_with_python(config_path, output_file, python_executable=None):
    if python_executable is None or str(python_executable) == sys.executable:
        return compute_aji_dice_from_outputs(config_path=config_path, output_file=output_file)

    command = [
        str(python_executable),
        str(Path(__file__).resolve()),
        "--internal-compute-aji-dice",
        "--config",
        str(config_path),
        "--output-file",
        str(output_file),
    ]
    env = build_runtime_env()
    completed = subprocess.run(command, check=True, env=env, capture_output=True, text=True)
    payload = None
    for line in reversed(completed.stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        if line.startswith("{") and line.endswith("}"):
            payload = json.loads(line)
            break
    if payload is None:
        raise RuntimeError(
            "Failed to parse AJI/Dice JSON from helper output. "
            f"stdout: {completed.stdout!r}, stderr: {completed.stderr!r}"
        )
    return {"aji": float(payload["aji"]), "dice": float(payload["dice"])}


def _infer_detection_config(experiment_dir):
    matches = sorted(experiment_dir.glob("*faster*rcnn*.py"))
    if len(matches) == 1:
        return matches[0]
    for candidate in matches:
        if "100e" in candidate.name:
            return candidate
    raise RuntimeError(f"Could not infer Faster R-CNN config in {experiment_dir}")


def _infer_mask_config(experiment_dir):
    matches = sorted(experiment_dir.glob("*mask*rcnn*.py"))
    if len(matches) == 1:
        return matches[0]
    for candidate in matches:
        if "100e" in candidate.name:
            return candidate
    raise RuntimeError(f"Could not infer Mask R-CNN config in {experiment_dir}")


def evaluate_detection_experiment(
    experiment_dir,
    config_path,
    select,
    extra_eval_options,
    low_mem=False,
    samples_per_gpu=1,
    workers_per_gpu=1,
    cuda_visible_devices=None,
    python_executable=None,
    bootstrap_resamples=1000,
    bootstrap_seed=42,
):
    experiment_dir = Path(experiment_dir)
    config_path = Path(config_path) if config_path else _infer_detection_config(experiment_dir)
    eval_dir = experiment_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = collect_checkpoints(experiment_dir, select=select)
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in {experiment_dir}")

    default_eval_options = _build_eval_option_pairs(extra_eval_options)
    runtime_cfg_options = build_runtime_cfg_options(
        low_mem=low_mem,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
    )
    _, dataset = _build_test_dataset(config_path)
    gt_dataset = _dataset_to_coco_gt_dict(dataset)

    records = []
    for checkpoint in checkpoints:
        ckpt_tag = checkpoint.stem
        run_dir = eval_dir / f"detection_{ckpt_tag}"
        run_dir.mkdir(parents=True, exist_ok=True)
        json_prefix = run_dir / "coco_results"
        run_detection_format_only(
            config_path=config_path,
            checkpoint_path=checkpoint,
            work_dir=run_dir,
            jsonfile_prefix=json_prefix,
            extra_eval_options=default_eval_options,
            cfg_options=runtime_cfg_options,
            low_mem=low_mem,
            cuda_visible_devices=cuda_visible_devices,
            python_executable=python_executable,
        )
        pred_bbox_file = _resolve_coco_result_json(json_prefix, iou_type="bbox")
        with open(pred_bbox_file, "r", encoding="utf-8") as f:
            pred_bbox = json.load(f)

        extracted = bootstrap_ci_from_coco_predictions(
            gt_dataset=gt_dataset,
            predictions=pred_bbox,
            iou_type="bbox",
            n_resamples=bootstrap_resamples,
            seed=bootstrap_seed,
            include_ap30=True,
            include_mar=True,
            progress_label=f"detection {ckpt_tag}",
        )

        record = {
            "checkpoint": str(checkpoint),
            "config": str(config_path),
            "metrics": extracted,
            "bootstrap": {
                "method": "percentile",
                "resampling_unit": "image",
                "confidence_level": 0.95,
                "n_resamples": int(bootstrap_resamples),
                "seed": int(bootstrap_seed),
            },
        }
        record_file = eval_dir / f"metrics_detection_{ckpt_tag}.json"
        with open(record_file, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        for path in [Path(str(json_prefix) + ".bbox.json"), Path(str(json_prefix) + ".segm.json"), Path(str(json_prefix) + ".json")]:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        try:
            run_dir.rmdir()
        except OSError:
            pass
        records.append(record)

    summary = {
        "task": "detection",
        "experiment_dir": str(experiment_dir),
        "config": str(config_path),
        "checkpoint_selection": select,
        "results": records,
    }
    summary_file = eval_dir / "metrics_detection_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary_file


def evaluate_segmentation_experiment(
    experiment_dir,
    config_path,
    select,
    extra_eval_options,
    low_mem=False,
    samples_per_gpu=1,
    workers_per_gpu=1,
    cuda_visible_devices=None,
    python_executable=None,
    bootstrap_resamples=1000,
    bootstrap_seed=42,
):
    experiment_dir = Path(experiment_dir)
    config_path = Path(config_path) if config_path else _infer_mask_config(experiment_dir)
    eval_dir = experiment_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = collect_checkpoints(experiment_dir, select=select)
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in {experiment_dir}")

    eval_options = _build_eval_option_pairs(extra_eval_options)
    runtime_cfg_options = build_runtime_cfg_options(
        low_mem=low_mem,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
    )
    _, dataset = _build_test_dataset(config_path)
    gt_dataset = _dataset_to_coco_gt_dict(dataset)

    records = []
    for checkpoint in checkpoints:
        ckpt_tag = checkpoint.stem
        run_dir = eval_dir / f"segmentation_{ckpt_tag}"
        json_prefix = run_dir / "coco_results"
        run_detection_format_only(
            config_path=config_path,
            checkpoint_path=checkpoint,
            work_dir=run_dir,
            jsonfile_prefix=json_prefix,
            extra_eval_options=eval_options,
            cfg_options=runtime_cfg_options,
            low_mem=low_mem,
            cuda_visible_devices=cuda_visible_devices,
            python_executable=python_executable,
        )
        pred_segm_file = _resolve_coco_result_json(json_prefix, iou_type="segm")
        with open(pred_segm_file, "r", encoding="utf-8") as f:
            pred_segm = json.load(f)

        coco_metrics = bootstrap_ci_from_coco_predictions(
            gt_dataset=gt_dataset,
            predictions=pred_segm,
            iou_type="segm",
            n_resamples=bootstrap_resamples,
            seed=bootstrap_seed,
            include_ap30=False,
            include_mar=False,
            progress_label=f"segmentation {ckpt_tag}",
        )

        aji_dice_values = compute_aji_dice_image_values_from_coco_dict(gt_dataset, pred_segm)
        ci_aji = bootstrap_ci_from_image_values(
            aji_dice_values["aji_values"],
            n_resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        )
        ci_dice = bootstrap_ci_from_image_values(
            aji_dice_values["dice_values"],
            n_resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        )
        merged_metrics = {
            "mAP": coco_metrics["mAP"],
            "AP50": coco_metrics["AP50"],
            "AJI": metric_with_bootstrap_values(aji_dice_values["aji_values"]),
            "Dice": metric_with_bootstrap_values(aji_dice_values["dice_values"]),
        }

        record = {
            "checkpoint": str(checkpoint),
            "config": str(config_path),
            "metrics": merged_metrics,
            "bootstrap": {
                "method": "percentile",
                "resampling_unit": "image",
                "confidence_level": 0.95,
                "n_resamples": int(bootstrap_resamples),
                "seed": int(bootstrap_seed),
            },
        }
        record_file = eval_dir / f"metrics_segmentation_{ckpt_tag}.json"
        with open(record_file, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        for path in [Path(str(json_prefix) + ".bbox.json"), Path(str(json_prefix) + ".segm.json"), Path(str(json_prefix) + ".json")]:
            try:
                Path(path).unlink()
            except FileNotFoundError:
                pass
        try:
            run_dir.rmdir()
        except OSError:
            pass
        records.append(record)

    summary = {
        "task": "segmentation",
        "experiment_dir": str(experiment_dir),
        "config": str(config_path),
        "checkpoint_selection": select,
        "results": records,
    }
    summary_file = eval_dir / "metrics_segmentation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary_file


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DINOv3 MMDet experiments with custom metrics")
    parser.add_argument("--internal-compute-aji-dice", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output-file", default=None, help=argparse.SUPPRESS)

    parser.add_argument("--task", choices=["detection", "segmentation"], default=None)
    parser.add_argument("--experiment-dir", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint-select", choices=["all", "latest", "best", "final"], default="all")
    parser.add_argument("--low-mem", action="store_true", help="Use lower-memory dataloader/runtime settings")
    parser.add_argument("--samples-per-gpu", type=int, default=1)
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--python-executable",
        default=None,
        help="Python executable to run detection/test.py (defaults to current interpreter)",
    )
    parser.add_argument(
        "--extra-eval-option",
        action="append",
        default=[],
        help="Additional evaluation option key=value; can be passed multiple times",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.internal_compute_aji_dice:
        metrics = compute_aji_dice_from_outputs(config_path=args.config, output_file=args.output_file)
        print(json.dumps(metrics))
        return
    if args.task is None:
        raise ValueError("--task is required unless using --internal-compute-aji-dice")
    if args.experiment_dir is None:
        raise ValueError("--experiment-dir is required for detection/segmentation evaluation")
    if args.task == "detection":
        summary_file = evaluate_detection_experiment(
            experiment_dir=args.experiment_dir,
            config_path=args.config,
            select=args.checkpoint_select,
            extra_eval_options=args.extra_eval_option,
            low_mem=args.low_mem,
            samples_per_gpu=args.samples_per_gpu,
            workers_per_gpu=args.workers_per_gpu,
            cuda_visible_devices=args.cuda_visible_devices,
            python_executable=args.python_executable,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        )
    else:
        summary_file = evaluate_segmentation_experiment(
            experiment_dir=args.experiment_dir,
            config_path=args.config,
            select=args.checkpoint_select,
            extra_eval_options=args.extra_eval_option,
            low_mem=args.low_mem,
            samples_per_gpu=args.samples_per_gpu,
            workers_per_gpu=args.workers_per_gpu,
            cuda_visible_devices=args.cuda_visible_devices,
            python_executable=args.python_executable,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        )
    print(str(summary_file))


if __name__ == "__main__":
    main()
