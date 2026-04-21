from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


def _point_in_polygon(xs, ys, polygon):
    inside = np.zeros_like(xs, dtype=bool)
    poly = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if poly.shape[0] < 3:
        return inside
    x0 = poly[:, 0]
    y0 = poly[:, 1]
    x1 = np.roll(x0, -1)
    y1 = np.roll(y0, -1)
    for i in range(poly.shape[0]):
        intersects = ((y0[i] > ys) != (y1[i] > ys))
        den = y1[i] - y0[i]
        den = np.where(np.abs(den) < 1e-8, 1e-8, den)
        x_inter = (x1[i] - x0[i]) * (ys - y0[i]) / den + x0[i]
        inside ^= intersects & (xs < x_inter)
    return inside


def polygons_to_mask(polygons, height, width):
    if height <= 0 or width <= 0:
        raise ValueError("invalid image shape for mask decoding")
    ys, xs = np.mgrid[0:height, 0:width]
    xs = xs.astype(np.float32) + 0.5
    ys = ys.astype(np.float32) + 0.5
    mask = np.zeros((height, width), dtype=bool)
    for polygon in polygons:
        if len(polygon) < 6:
            continue
        mask |= _point_in_polygon(xs, ys, polygon)
    return mask


def decode_segmentation(segmentation, height, width):
    if isinstance(segmentation, list):
        return polygons_to_mask(segmentation, height, width)
    if isinstance(segmentation, dict):
        counts = segmentation.get("counts")
        size = segmentation.get("size")
        if counts is None or size is None:
            raise ValueError("segmentation dict must contain 'counts' and 'size'")
        if not isinstance(size, (list, tuple)) or len(size) != 2:
            raise ValueError("segmentation 'size' must be a [height, width] pair")
        if int(size[0]) != int(height) or int(size[1]) != int(width):
            got_h, got_w = int(size[0]), int(size[1])
            raise ValueError(
                f"segmentation size mismatch: expected {[height, width]}, got {[got_h, got_w]}"
            )
        try:
            if isinstance(counts, list):
                rle = mask_utils.frPyObjects(segmentation, height, width)
            elif isinstance(counts, (str, bytes)):
                rle = {"counts": counts, "size": [height, width]}
            else:
                raise ValueError("segmentation 'counts' must be list, str, or bytes")
            mask = mask_utils.decode(rle)
        except Exception as exc:  # pragma: no cover - payload-dependent decoder errors
            raise ValueError("malformed segmentation payload") from exc
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.shape != (height, width):
            raise ValueError(
                f"decoded segmentation shape mismatch: expected {(height, width)}, got {tuple(mask.shape)}"
            )
        return mask.astype(bool)
    raise ValueError(f"unsupported segmentation encoding: {type(segmentation).__name__}")


@DATASETS.register_module()
class CocoInstanceDatasetBridge(CustomDataset):
    CLASSES = ("lesion",)
    PALETTE = [[255, 0, 0]]

    @staticmethod
    def _validate_category_schema(categories, expected_categories):
        expected_names = list(expected_categories)
        actual_names = [str(cat.get("name", "")) for cat in categories]
        if actual_names != expected_names:
            raise ValueError(
                "invalid COCO category schema: expected exactly "
                f"{expected_names}, got {actual_names}"
            )
        actual_ids = [int(cat.get("id", -1)) for cat in categories]
        if len(set(actual_ids)) != len(actual_ids):
            raise ValueError(
                "invalid COCO category schema: category ids must be unique; "
                f"got {actual_ids}"
            )

    def __init__(self, img_suffix=".png", ann_suffix=".json", ann_file=None, expected_categories=None, **kwargs):
        self.ann_suffix = ann_suffix
        self.ann_file = ann_file
        self.expected_categories = tuple(expected_categories) if expected_categories else tuple(self.CLASSES)
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=".png",
            reduce_zero_label=False,
            **kwargs,
        )

    @staticmethod
    def _load_coco_annotations(ann_file, expected_categories):
        ann_path = Path(ann_file)
        payload = json.loads(ann_path.read_text())
        categories = payload.get("categories", [])
        CocoInstanceDatasetBridge._validate_category_schema(categories, expected_categories)
        cat_id_map = {cat["id"]: idx for idx, cat in enumerate(categories)}
        images = payload.get("images", [])
        ann_by_image = {}
        for ann in payload.get("annotations", []):
            ann_by_image.setdefault(ann["image_id"], []).append(ann)

        data_infos = []
        for image_info in images:
            height = int(image_info["height"])
            width = int(image_info["width"])
            img_id = image_info["id"]
            per_image_anns = ann_by_image.get(img_id, [])

            data_infos.append(
                {
                    "id": img_id,
                    "filename": image_info["file_name"],
                    "height": height,
                    "width": width,
                    "ann": {
                        "height": height,
                        "width": width,
                        "cat_id_map": cat_id_map,
                        "raw_annotations": per_image_anns,
                    },
                }
            )
        return data_infos

    @staticmethod
    def build_dense_targets(ann_info):
        height = int(ann_info["height"])
        width = int(ann_info["width"])
        cat_id_map = ann_info["cat_id_map"]
        raw_annotations = ann_info.get("raw_annotations", [])

        gt_labels = []
        gt_masks = []
        gt_semantic_seg = np.full((height, width), 255, dtype=np.int64)
        gt_instance_map = np.zeros((height, width), dtype=np.int64)
        inst_id = 1

        for ann in raw_annotations:
            if int(ann.get("iscrowd", 0)) == 1:
                continue
            cat_id = ann.get("category_id")
            if cat_id not in cat_id_map:
                continue
            segmentation = ann.get("segmentation", [])
            try:
                mask = decode_segmentation(segmentation, height, width)
            except ValueError as exc:
                raise ValueError(
                    f"segmentation decode failure for annotation id {ann.get('id')}: {exc}"
                ) from exc
            if not np.any(mask):
                continue
            label = cat_id_map[cat_id]
            gt_labels.append(label)
            gt_masks.append(mask.astype(np.int64))
            # Deterministic overlap policy: later annotations overwrite earlier ones.
            gt_semantic_seg[mask] = label
            gt_instance_map[mask] = inst_id
            inst_id += 1

        if gt_masks:
            stacked_masks = np.stack(gt_masks, axis=0).astype(np.int64)
            labels_arr = np.asarray(gt_labels, dtype=np.int64)
        else:
            stacked_masks = np.empty((0, height, width), dtype=np.int64)
            labels_arr = np.empty((0,), dtype=np.int64)

        return {
            "gt_labels": labels_arr,
            "gt_masks": stacked_masks,
            "gt_semantic_seg": gt_semantic_seg,
            "gt_instance_map": gt_instance_map,
        }

    def load_annotations(self, img_dir=None, img_suffix=None, ann_dir=None, seg_map_suffix=None, split=None):
        ann_file = self.ann_file or ann_dir
        if ann_file is None:
            raise ValueError('ann_file or ann_dir must point to a COCO json file')
        return self._load_coco_annotations(ann_file, self.expected_categories)

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann']

    @classmethod
    def load_annotations_from_file(cls, ann_file, expected_categories=None):
        expected = tuple(expected_categories) if expected_categories else tuple(cls.CLASSES)
        return cls._load_coco_annotations(ann_file, expected)
