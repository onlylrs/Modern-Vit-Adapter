from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

try:
    from safetensors.torch import load_file as _load_file
except ModuleNotFoundError:  # pragma: no cover - exercised in older envs
    _load_file = None


@dataclass(frozen=True)
class DINOv3CheckpointInfo:
    config_path: Path
    embed_dim: int
    depth: int
    num_heads: int
    patch_size: int
    num_register_tokens: int


def _config_path(root: Path) -> Path:
    return root / "config.json"


def read_dinov3_metadata(root: Path | str) -> DINOv3CheckpointInfo:
    root_path = Path(root)
    config_path = _config_path(root_path)
    try:
        data = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError("Malformed DINOv3 config: invalid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("Malformed DINOv3 config: expected a JSON object")
    required_keys = (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "patch_size",
    )
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Malformed DINOv3 config: missing required key '{key}'")
    try:
        embed_dim = int(data["hidden_size"])
        depth = int(data["num_hidden_layers"])
        num_heads = int(data["num_attention_heads"])
        patch_size = int(data["patch_size"])
        num_register_tokens = int(data.get("num_register_tokens", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("Malformed DINOv3 config: required keys must be an integer") from exc
    if embed_dim <= 0:
        raise ValueError("Malformed DINOv3 config: hidden_size must be positive")
    if depth <= 0:
        raise ValueError("Malformed DINOv3 config: num_hidden_layers must be positive")
    if num_heads <= 0:
        raise ValueError("Malformed DINOv3 config: num_attention_heads must be positive")
    if patch_size <= 0:
        raise ValueError("Malformed DINOv3 config: patch_size must be positive")
    if num_register_tokens < 0:
        raise ValueError("Malformed DINOv3 config: num_register_tokens must be non-negative")
    return DINOv3CheckpointInfo(
        config_path=config_path,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        patch_size=patch_size,
        num_register_tokens=num_register_tokens,
    )


def validate_dinov3_variant(
    *, embed_dim: int, depth: int, patch_size: int, num_heads: int
) -> None:
    if (embed_dim, depth, patch_size) != (768, 12, 16):
        raise ValueError("Only ViT-B/16 DINOv3 checkpoints are supported")
    if num_heads != 12:
        raise ValueError("Only ViT-B/16 DINOv3 checkpoints with 12 heads are supported")


def load_dinov3_checkpoint(root: Path | str) -> Tuple[DINOv3CheckpointInfo, Dict[str, torch.Tensor]]:
    root_path = Path(root)
    info = read_dinov3_metadata(root_path)
    validate_dinov3_variant(
        embed_dim=info.embed_dim,
        depth=info.depth,
        patch_size=info.patch_size,
        num_heads=info.num_heads,
    )
    state_dict = _load_safetensors(root_path / "model.safetensors")
    return info, state_dict


def freeze_module(module: torch.nn.Module) -> torch.nn.Module:
    for parameter in module.parameters():
        parameter.requires_grad = False
    return module


def _load_safetensors(path: Path) -> Dict[str, torch.Tensor]:
    _validate_safetensors_structure(path)
    if _load_file is not None:
        return _load_file(str(path))
    return _load_safetensors_fallback(path)


def _validate_safetensors_structure(path: Path) -> None:
    data = path.read_bytes()
    if len(data) < 8:
        raise ValueError("Malformed safetensors file: missing header length")
    header_size = int.from_bytes(data[:8], "little")
    if header_size <= 0 or 8 + header_size > len(data):
        raise ValueError("Malformed safetensors file: invalid header length")
    try:
        header = json.loads(data[8 : 8 + header_size])
    except json.JSONDecodeError as exc:
        raise ValueError("Malformed safetensors file: invalid JSON header") from exc
    if not isinstance(header, dict):
        raise ValueError("Malformed safetensors file: header must be a JSON object")
    if "__metadata__" in header and not isinstance(header["__metadata__"], dict):
        raise ValueError("Malformed safetensors file: __metadata__ must be an object")

    ranges = []
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(spec, dict):
            raise ValueError(f"Malformed safetensors file: tensor '{name}' spec must be an object")
        if "dtype" not in spec or "shape" not in spec or "data_offsets" not in spec:
            raise ValueError(f"Malformed safetensors file: tensor '{name}' is missing required fields")
        dtype_name = spec["dtype"]
        if dtype_name not in _SAFETENSORS_DTYPES:
            raise ValueError(f"Malformed safetensors file: unsupported dtype '{dtype_name}'")
        shape = spec["shape"]
        if not isinstance(shape, list) or any(not isinstance(dim, int) or dim < 0 for dim in shape):
            raise ValueError(f"Malformed safetensors file: tensor '{name}' has invalid shape")
        offsets = spec["data_offsets"]
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"Malformed safetensors file: tensor '{name}' has invalid offsets")
        start, end = offsets
        if not all(isinstance(value, int) for value in (start, end)):
            raise ValueError(f"Malformed safetensors file: tensor '{name}' offsets must be integers")
        if start < 0 or end < start or end > len(data) - (8 + header_size):
            raise ValueError(f"Malformed safetensors file: tensor '{name}' offsets out of bounds")
        itemsize = _ITEMSIZE_BY_DTYPE[dtype_name]
        if (end - start) % itemsize != 0:
            raise ValueError(f"Malformed safetensors file: tensor '{name}' byte size is inconsistent")
        expected_items = 1
        for dim in shape:
            expected_items *= dim
        if expected_items * itemsize != end - start:
            raise ValueError(f"Malformed safetensors file: tensor '{name}' byte size does not match shape")
        ranges.append((start, end, name))

    ranges.sort()
    previous_end = 0
    for start, end, name in ranges:
        if start < previous_end:
            raise ValueError(f"Malformed safetensors file: tensor '{name}' offsets overlap")
        previous_end = end
    payload_size = len(data) - (8 + header_size)
    if previous_end != payload_size:
        raise ValueError("Malformed safetensors file: trailing unused payload bytes")


def _load_safetensors_fallback(path: Path) -> Dict[str, torch.Tensor]:
    data = path.read_bytes()
    header_size = int.from_bytes(data[:8], "little")
    header = json.loads(data[8 : 8 + header_size])
    payload = memoryview(data)[8 + header_size :]

    tensors: Dict[str, torch.Tensor] = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        dtype_name = spec["dtype"]
        start, end = spec["data_offsets"]
        raw = payload[start:end].tobytes()
        if dtype_name == "BF16":
            array = np.frombuffer(raw, dtype=np.uint16).copy()
            tensor = torch.from_numpy(array.view(np.int16)).view(torch.bfloat16)
        else:
            array = np.frombuffer(raw, dtype=_NUMPY_DTYPES[dtype_name]).copy()
            tensor = torch.from_numpy(array).to(_SAFETENSORS_DTYPES[dtype_name])
        tensors[name] = tensor.reshape(tuple(spec["shape"]))
    return tensors


_SAFETENSORS_DTYPES = {
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "BF16": torch.bfloat16,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


_ITEMSIZE_BY_DTYPE = {
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "BF16": 2,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "U8": 1,
    "BOOL": 1,
}

_NUMPY_DTYPES = {
    "F16": np.float16,
    "F32": np.float32,
    "F64": np.float64,
    "BF16": np.uint16,
    "I8": np.int8,
    "I16": np.int16,
    "I32": np.int32,
    "I64": np.int64,
    "U8": np.uint8,
    "BOOL": np.bool_,
}
