from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from transformers import AutoModel


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
    def from_checkpoint(cls, checkpoint_root: Path | str) -> "OfficialDINOv3Backbone":
        root = Path(checkpoint_root)
        model = AutoModel.from_pretrained(str(root), local_files_only=True, trust_remote_code=False)
        model.eval()
        return cls(model=model, checkpoint_root=root)

    def forward(self, pixel_values: torch.Tensor):
        return self.model(pixel_values=pixel_values)

    @staticmethod
    def _resolve_layers(model: nn.Module):
        if hasattr(model, "layer"):
            return model.layer
        if hasattr(model, "model") and hasattr(model.model, "layer"):
            return model.model.layer
        raise AttributeError(f"{type(model).__name__} does not expose transformer layers")

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
