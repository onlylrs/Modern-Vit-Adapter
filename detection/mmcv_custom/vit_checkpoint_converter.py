from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F


def load_vit_checkpoint(filename, map_location=None, checkpoint_format="auto", target_state=None, target_format="timm"):
    path = Path(filename)
    if path.is_dir():
        raw = _load_directory_checkpoint(path, map_location=map_location)
        return normalize_vit_state_dict(
            raw,
            checkpoint_format=checkpoint_format,
            target_state=target_state,
            target_format=target_format,
        )
    raw = torch.load(str(path), map_location=map_location)
    return normalize_vit_state_dict(
        raw,
        checkpoint_format=checkpoint_format,
        target_state=target_state,
        target_format=target_format,
    )


def normalize_vit_state_dict(raw_checkpoint, *, checkpoint_format="auto", target_state=None, target_format="timm"):
    state_dict = _extract_state_dict(raw_checkpoint, checkpoint_format)
    checkpoint_format = _infer_format(state_dict, checkpoint_format)
    converted = OrderedDict()
    for key, value in state_dict.items():
        if _drop_key(key, target_format=target_format):
            continue
        key = _strip_known_prefixes(key)
        if _drop_key(key, target_format=target_format):
            continue
        if checkpoint_format == "dinov3":
            _append_dinov3_key(converted, key, value, target_format=target_format)
        else:
            converted[_normalize_timm_key(key)] = value
    if target_state is not None:
        converted = _adapt_to_target_shapes(converted, target_state)
    return converted


def _load_directory_checkpoint(path: Path, map_location=None):
    safetensors_path = path / "model.safetensors"
    if safetensors_path.is_file():
        try:
            from safetensors.torch import load_file
        except ModuleNotFoundError as exc:
            raise RuntimeError("safetensors is required to load directory checkpoints") from exc
        return load_file(str(safetensors_path), device="cpu")
    bin_path = path / "pytorch_model.bin"
    if bin_path.is_file():
        raw = torch.load(str(bin_path), map_location=map_location)
        return normalize_vit_state_dict(raw)
    raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in {path}")


def _extract_state_dict(raw_checkpoint, checkpoint_format):
    if not isinstance(raw_checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")
    if checkpoint_format in {"ccs", "smartcyto"}:
        raw_checkpoint = raw_checkpoint["teacher"]
    elif checkpoint_format in {"auto", "dinov3"} and "teacher" in raw_checkpoint and isinstance(raw_checkpoint["teacher"], Mapping):
        raw_checkpoint = raw_checkpoint["teacher"]
    elif "state_dict" in raw_checkpoint and isinstance(raw_checkpoint["state_dict"], Mapping):
        raw_checkpoint = raw_checkpoint["state_dict"]
    elif "model" in raw_checkpoint and isinstance(raw_checkpoint["model"], Mapping):
        raw_checkpoint = raw_checkpoint["model"]
    return raw_checkpoint


def _infer_format(state_dict, checkpoint_format):
    if checkpoint_format != "auto":
        return checkpoint_format
    keys = list(state_dict.keys())
    if any("rope_embed" in key or "storage_tokens" in key or "attn.qkv.bias_mask" in key for key in keys):
        return "dinov3"
    return "timm"


def _strip_known_prefixes(key):
    for prefix in ("module.", "teacher.", "backbone.", "model."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def _drop_key(key, target_format="timm"):
    dropped_prefixes = ("dino_head.", "ibot_head.", "head.")
    if target_format != "hf":
        dropped_prefixes = dropped_prefixes + ("mask_token",)
    return key.startswith(dropped_prefixes) or key in {
        "rope_embed.periods",
    }


def _flatten_grouped_block_key(key):
    match = re.match(r"blocks\.(\d+)\.(\d+)\.(.+)", key)
    if match is None:
        return key
    return f"blocks.{int(match.group(2))}.{match.group(3)}"


def _normalize_timm_key(key):
    key = _flatten_grouped_block_key(key)
    key = key.replace(".ls1.gamma", ".gamma1")
    key = key.replace(".ls2.gamma", ".gamma2")
    return key


def _append_dinov3_key(converted, key, value, target_format="timm"):
    if target_format == "hf":
        _append_dinov3_hf_key(converted, key, value)
        return

    if key in {"storage_tokens", "register_tokens"}:
        converted["register_tokens"] = value
        return
    if key.endswith("attn.qkv.bias_mask") or key == "rope_embed.periods":
        return
    qkv_match = re.match(r"blocks\.(\d+)\.attn\.qkv\.(weight|bias)", key)
    if qkv_match:
        block_idx, suffix = qkv_match.groups()
        q, k, v = value.chunk(3, dim=0)
        converted[f"blocks.{block_idx}.attn.q_proj.{suffix}"] = q
        if suffix == "weight":
            converted[f"blocks.{block_idx}.attn.k_proj.{suffix}"] = k
        converted[f"blocks.{block_idx}.attn.v_proj.{suffix}"] = v
        return
    replacements = (
        (".attn.proj.", ".attn.o_proj."),
        (".mlp.fc1.", ".mlp.up_proj."),
        (".mlp.fc2.", ".mlp.down_proj."),
        (".ls1.gamma", ".layer_scale1"),
        (".ls2.gamma", ".layer_scale2"),
    )
    for old, new in replacements:
        if old in key:
            key = key.replace(old, new)
            break
    converted[key] = value


def _append_dinov3_hf_key(converted, key, value):
    if key == "cls_token":
        converted["embeddings.cls_token"] = value
        return
    if key == "mask_token":
        converted["embeddings.mask_token"] = value.unsqueeze(0) if value.ndim == 2 else value
        return
    if key in {"storage_tokens", "register_tokens"}:
        converted["embeddings.register_tokens"] = value
        return
    if key == "patch_embed.proj.weight":
        converted["embeddings.patch_embeddings.weight"] = value
        return
    if key == "patch_embed.proj.bias":
        converted["embeddings.patch_embeddings.bias"] = value
        return
    if key in {"rope_embed.periods"} or key.endswith("attn.qkv.bias_mask"):
        return

    qkv_match = re.match(r"blocks\.(\d+)\.attn\.qkv\.(weight|bias)", key)
    if qkv_match:
        block_idx, suffix = qkv_match.groups()
        q, k, v = value.chunk(3, dim=0)
        prefix = f"layer.{block_idx}.attention"
        converted[f"{prefix}.q_proj.{suffix}"] = q
        if suffix == "weight":
            converted[f"{prefix}.k_proj.{suffix}"] = k
        converted[f"{prefix}.v_proj.{suffix}"] = v
        return

    if key.startswith("blocks."):
        key = key.replace("blocks.", "layer.", 1)
    replacements = (
        (".attn.proj.", ".attention.o_proj."),
        (".mlp.fc1.", ".mlp.up_proj."),
        (".mlp.fc2.", ".mlp.down_proj."),
        (".ls1.gamma", ".layer_scale1.lambda1"),
        (".ls2.gamma", ".layer_scale2.lambda1"),
    )
    for old, new in replacements:
        if old in key:
            key = key.replace(old, new)
            break
    converted[key] = value


def _adapt_to_target_shapes(state_dict, target_state):
    adapted = OrderedDict()
    for key, value in state_dict.items():
        target = target_state.get(key)
        if target is None:
            adapted[key] = value
            continue
        if tuple(value.shape) == tuple(target.shape):
            adapted[key] = value
        elif key == "patch_embed.proj.weight" and value.ndim == 4 and target.ndim == 4:
            adapted[key] = F.interpolate(value, size=target.shape[-2:], mode="bicubic", align_corners=False)
        elif key == "pos_embed" and value.ndim == 3 and target.ndim == 3:
            adapted[key] = _resize_pos_embed(value, target)
        else:
            adapted[key] = value
    return adapted


def _resize_pos_embed(value, target):
    if value.shape[1] == target.shape[1]:
        return value
    cls_value, patch_value = value[:, :1], value[:, 1:]
    target_cls_tokens = 1
    target_patch_count = target.shape[1] - target_cls_tokens
    src_size = int(patch_value.shape[1] ** 0.5)
    dst_size = int(target_patch_count ** 0.5)
    if src_size * src_size != patch_value.shape[1] or dst_size * dst_size != target_patch_count:
        return value
    patch_value = patch_value.reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2)
    patch_value = F.interpolate(patch_value, size=(dst_size, dst_size), mode="bicubic", align_corners=False)
    patch_value = patch_value.reshape(1, -1, target_patch_count).permute(0, 2, 1)
    return torch.cat((cls_value, patch_value), dim=1)
