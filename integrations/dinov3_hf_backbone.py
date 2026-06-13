from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from transformers import AutoModel, DINOv3ViTConfig, DINOv3ViTModel


class OfficialDINOv3Backbone(nn.Module):
    def __init__(self, model: nn.Module, checkpoint_root: Path) -> None:
        super().__init__()
        self.model = model
        self.checkpoint_root = Path(checkpoint_root)
        self.patch_size = int(self.model.config.patch_size)
        self.embed_dim = int(self.model.config.hidden_size)
        self.layers = self._resolve_layers(self.model)
        self.n_blocks = len(self.layers)
        self.n_storage_tokens = int(getattr(self.model.config, "num_register_tokens", 0))

    @classmethod
    def from_checkpoint(cls, checkpoint_root: Path | str, checkpoint_format: str = "auto") -> "OfficialDINOv3Backbone":
        root = Path(checkpoint_root)
        if root.is_dir():
            model = AutoModel.from_pretrained(str(root), local_files_only=True, trust_remote_code=False)
        elif root.is_file():
            model = _load_raw_dinov3_model(root, checkpoint_format=checkpoint_format)
        else:
            raise FileNotFoundError(f"DINOv3 checkpoint not found: {root}")
        model.eval()
        return cls(model=model, checkpoint_root=root)

    def forward(self, pixel_values: torch.Tensor):
        return self.model(pixel_values=pixel_values)

    @property
    def num_prefix_tokens(self) -> int:
        return 1 + self.n_storage_tokens

    @staticmethod
    def _resolve_layers(model: nn.Module):
        if hasattr(model, "layer"):
            return model.layer
        if hasattr(model, "model") and hasattr(model.model, "layer"):
            return model.model.layer
        raise AttributeError(f"{type(model).__name__} does not expose transformer layers")

    def prepare_tokens(self, x: torch.Tensor):
        hidden_states = self.model.embeddings(pixel_values=x)
        position_embeddings = self.model.rope_embeddings(x)
        return hidden_states, position_embeddings

    def run_layers(self, hidden_states: torch.Tensor, position_embeddings: torch.Tensor, start: int, end: int):
        if end < start:
            return hidden_states
        if start < 0 or end >= self.n_blocks:
            raise ValueError(f"Invalid DINOv3 layer range [{start}, {end}] for {self.n_blocks} blocks")
        for idx in range(start, end + 1):
            hidden_states = self.layers[idx](hidden_states, position_embeddings=position_embeddings)
        return hidden_states

    def split_tokens(self, hidden_states: torch.Tensor, norm: bool = True):
        if norm:
            hidden_states = self.model.norm(hidden_states)
        class_token = hidden_states[:, 0]
        extra_tokens = hidden_states[:, 1 : self.n_storage_tokens + 1]
        patch_tokens = hidden_states[:, self.n_storage_tokens + 1 :]
        return patch_tokens, class_token, extra_tokens

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        *,
        n: int | Sequence[int] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ):
        hidden_states = self.model.embeddings(pixel_values=x)
        position_embeddings = self.model.rope_embeddings(x)

        if isinstance(n, int):
            blocks_to_take = list(range(self.n_blocks - n, self.n_blocks))
        else:
            blocks_to_take = list(n)

        outputs = []
        for idx, layer_module in enumerate(self.layers):
            hidden_states = layer_module(hidden_states, position_embeddings=position_embeddings)
            if idx in blocks_to_take:
                outputs.append(hidden_states)

        if norm:
            outputs = [self.model.norm(out) for out in outputs]

        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1 : self.n_storage_tokens + 1] for out in outputs]
        patch_tokens = [out[:, self.n_storage_tokens + 1 :] for out in outputs]

        if reshape:
            batch_size, _, height, width = x.shape
            patch_tokens = [
                out.reshape(batch_size, height // self.patch_size, width // self.patch_size, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in patch_tokens
            ]

        if not return_class_token and not return_extra_tokens:
            return tuple(patch_tokens)
        if return_class_token and not return_extra_tokens:
            return tuple(zip(patch_tokens, class_tokens))
        if not return_class_token and return_extra_tokens:
            return tuple(zip(patch_tokens, extra_tokens))
        return tuple(zip(patch_tokens, class_tokens, extra_tokens))


def normalize_interaction_indexes(interaction_indexes):
    normalized = []
    for item in interaction_indexes:
        if isinstance(item, (list, tuple)):
            if not item:
                raise ValueError("interaction index group cannot be empty")
            normalized.append(int(item[-1]))
        else:
            normalized.append(int(item))
    return normalized


def normalize_interaction_ranges(interaction_indexes):
    ranges = []
    next_start = 0
    for item in interaction_indexes:
        if isinstance(item, (list, tuple)):
            if not item:
                raise ValueError("interaction index group cannot be empty")
            start, end = int(item[0]), int(item[-1])
        else:
            start, end = next_start, int(item)
        if start < 0 or end < start:
            raise ValueError(f"invalid interaction index range: [{start}, {end}]")
        ranges.append((start, end))
        next_start = end + 1
    return ranges


def _load_raw_dinov3_model(checkpoint_path: Path, checkpoint_format: str = "auto") -> nn.Module:
    config = _build_config_from_raw_checkpoint(checkpoint_path)
    model = DINOv3ViTModel(config)
    state_dict = _load_vit_checkpoint(
        str(checkpoint_path),
        map_location="cpu",
        checkpoint_format=checkpoint_format,
        target_format="hf",
    )
    state_dict = _align_hf_state_dict_prefixes(state_dict, model.state_dict())
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = [
        key for key in incompatible.missing_keys
        if key not in {"pooler.dense.weight", "pooler.dense.bias"}
    ]
    if missing:
        raise RuntimeError(f"Missing DINOv3 keys when loading {checkpoint_path}: {missing[:20]}")
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected DINOv3 keys when loading {checkpoint_path}: {incompatible.unexpected_keys[:20]}")
    return model


def _load_vit_checkpoint(*args, **kwargs):
    converter_path = Path(__file__).resolve().parents[1] / "detection" / "mmcv_custom" / "vit_checkpoint_converter.py"
    spec = importlib.util.spec_from_file_location("_modern_vit_checkpoint_converter", converter_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load DINOv3 checkpoint converter from {converter_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_vit_checkpoint(*args, **kwargs)


def _align_hf_state_dict_prefixes(state_dict, target_state):
    if any(key.startswith("model.layer.") for key in target_state):
        state_dict = {
            (f"model.{key}" if key.startswith("layer.") else key): value
            for key, value in state_dict.items()
        }
    if "model.norm.weight" in target_state and "norm.weight" in state_dict:
        state_dict["model.norm.weight"] = state_dict.pop("norm.weight")
    if "model.norm.bias" in target_state and "norm.bias" in state_dict:
        state_dict["model.norm.bias"] = state_dict.pop("norm.bias")
    return state_dict


def _build_config_from_raw_checkpoint(checkpoint_path: Path) -> DINOv3ViTConfig:
    state_dict = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(state_dict, dict) and "teacher" in state_dict and isinstance(state_dict["teacher"], dict):
        state_dict = state_dict["teacher"]
    if not isinstance(state_dict, dict):
        raise TypeError(f"DINOv3 checkpoint must contain a state dict: {checkpoint_path}")

    def tensor_shape(key: str):
        for candidate in (key, f"backbone.{key}", f"module.{key}", f"model.{key}"):
            tensor = state_dict.get(candidate)
            if tensor is not None:
                return tuple(tensor.shape)
        raise KeyError(f"Missing required DINOv3 key in {checkpoint_path}: {key}")

    def has_block_norm_key(key: str) -> bool:
        return key.startswith("blocks.") and key.endswith("norm1.weight")

    def has_backbone_block_norm_key(key: str) -> bool:
        return key.startswith("backbone.blocks.") and key.endswith("norm1.weight")

    embed_dim = tensor_shape("cls_token")[-1]
    patch_size = tensor_shape("patch_embed.proj.weight")[-1]
    depth = sum(1 for key in state_dict if has_block_norm_key(key))
    if depth == 0:
        depth = sum(1 for key in state_dict if has_backbone_block_norm_key(key))
    if depth == 0:
        raise ValueError(f"Cannot infer DINOv3 depth from {checkpoint_path}")

    qkv_rows = tensor_shape("blocks.0.attn.qkv.weight")[0]
    num_heads = 16 if embed_dim == 1024 else max(1, embed_dim // 64)
    if qkv_rows % (3 * num_heads) != 0:
        raise ValueError(f"Cannot infer DINOv3 head count from {checkpoint_path}")

    num_register_tokens = tensor_shape("storage_tokens")[1]
    return DINOv3ViTConfig(
        hidden_size=embed_dim,
        num_hidden_layers=depth,
        num_attention_heads=num_heads,
        intermediate_size=embed_dim * 4,
        patch_size=patch_size,
        image_size=224,
        num_register_tokens=num_register_tokens,
        layerscale_value=1.0,
    )
