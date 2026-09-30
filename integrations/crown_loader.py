"""Load the unmodified CROWN ViT-L/16 implementation and checkpoint."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import torch


def load_crown(checkpoint: str | Path, official_root: str | Path):
    root = Path(official_root).expanduser().resolve()
    if not (root / "models" / "vision_transformer.py").is_file():
        raise FileNotFoundError(f"CROWN official source not found: {root}")
    path = Path(checkpoint).expanduser()
    if path.is_dir():
        path = path / "CROWN.pth"
    if not path.is_file():
        raise FileNotFoundError(f"CROWN checkpoint not found: {path}")

    # The official source imports its own modules as `dinov2.*` although its
    # checkout is named CROWN. Register that checkout as the package here.
    package = sys.modules.get("dinov2")
    if package is not None and Path(package.__file__).resolve().parent != root:
        raise ImportError("A different dinov2 package is already imported; CROWN needs its official source")
    if package is None:
        spec = importlib.util.spec_from_file_location(
            "dinov2", root / "__init__.py", submodule_search_locations=[str(root)]
        )
        package = importlib.util.module_from_spec(spec)
        sys.modules["dinov2"] = package
        spec.loader.exec_module(package)

    vit_large = importlib.import_module("dinov2.models.vision_transformer").vit_large
    model = vit_large(
        patch_size=16, img_size=224, init_values=1.0,
        block_chunks=4, ffn_layer="swiglufused",
    )
    # Memory-mapping a checkpoint on sshfs can SIGBUS if the mount drops while
    # tensors are copied to a GPU. The experiment runner stages it locally;
    # regular loading also keeps manual use safe from that failure mode.
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True, assign=True)
    return model
