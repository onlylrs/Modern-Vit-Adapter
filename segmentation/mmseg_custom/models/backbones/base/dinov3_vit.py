from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from timm.models.layers import DropPath, to_2tuple


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        h, w = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return x, h, w


class DINOv3MLP(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.up_proj = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.down_proj = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.up_proj(x)
        x = self.act(x)
        x = self.down_proj(x)
        x = self.drop(x)
        return x


def _rotate_half(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    x_rot = torch.stack((-x2, x1), dim=-1)
    return x_rot.flatten(-2)


def _apply_rope(x, cos, sin):
    return (x * cos) + (_rotate_half(x) * sin)


class DINOv3Attention(nn.Module):
    def __init__(self, dim, num_heads, rope_theta=100.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("embed dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 4 != 0:
            raise ValueError("head_dim must be divisible by 4 for 2D RoPE")

        self.scale = self.head_dim ** -0.5
        self.rope_theta = float(rope_theta)
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.o_proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def _rope_cos_sin(self, h, w, device, dtype):
        if h * w <= 0:
            raise ValueError("Invalid patch grid for RoPE")
        half = self.head_dim // 2
        quarter = half // 2
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, quarter, device=device, dtype=torch.float32) / max(quarter, 1))
        )

        y = torch.arange(h, device=device, dtype=torch.float32).repeat_interleave(w)
        x = torch.arange(w, device=device, dtype=torch.float32).repeat(h)
        y_freqs = torch.einsum("n,d->nd", y, inv_freq)
        x_freqs = torch.einsum("n,d->nd", x, inv_freq)

        y_angles = torch.repeat_interleave(y_freqs, repeats=2, dim=-1)
        x_angles = torch.repeat_interleave(x_freqs, repeats=2, dim=-1)
        angles = torch.cat([y_angles, x_angles], dim=-1)
        cos = angles.cos().to(dtype=dtype)
        sin = angles.sin().to(dtype=dtype)
        return cos[None, None, :, :], sin[None, None, :, :]

    def forward(self, x, h, w, num_prefix_tokens=0):
        bsz, seq_len, dim = x.shape
        q = self.q_proj(x).reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if num_prefix_tokens < 0 or num_prefix_tokens > seq_len:
            raise ValueError("num_prefix_tokens is out of range")
        patch_len = seq_len - num_prefix_tokens
        if patch_len != h * w:
            raise ValueError("Patch token count must match H*W for RoPE")

        if patch_len > 0:
            cos, sin = self._rope_cos_sin(h, w, q.device, q.dtype)
            q_patch = _apply_rope(q[:, :, num_prefix_tokens:, :], cos, sin)
            k_patch = _apply_rope(k[:, :, num_prefix_tokens:, :], cos, sin)
            if num_prefix_tokens > 0:
                q = torch.cat([q[:, :, :num_prefix_tokens, :], q_patch], dim=2)
                k = torch.cat([k[:, :, :num_prefix_tokens, :], k_patch], dim=2)
            else:
                q, k = q_patch, k_patch

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(bsz, seq_len, dim)
        x = self.o_proj(x)
        x = self.proj_drop(x)
        return x


class DINOv3Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        rope_theta=100.0,
        with_cp=False,
        layer_scale_init=1.0,
    ):
        super().__init__()
        self.with_cp = with_cp
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DINOv3Attention(
            dim=dim,
            num_heads=num_heads,
            rope_theta=rope_theta,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = DINOv3MLP(dim=dim, hidden_dim=int(dim * mlp_ratio), drop=drop)
        self.layer_scale1 = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.layer_scale2 = nn.Parameter(layer_scale_init * torch.ones(dim))

    def forward(self, x, h, w, num_prefix_tokens=0):
        def _inner_forward(x):
            x = x + self.drop_path(
                self.layer_scale1 * self.attn(self.norm1(x), h, w, num_prefix_tokens=num_prefix_tokens)
            )
            x = x + self.drop_path(self.layer_scale2 * self.mlp(self.norm2(x)))
            return x

        if self.with_cp and x.requires_grad:
            return cp.checkpoint(_inner_forward, x)
        return _inner_forward(x)


class DINOv3VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        rope_theta=100.0,
        num_register_tokens=4,
        with_cp=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_register_tokens = int(num_register_tokens)
        self.rope_theta = float(rope_theta)
        self.norm_layer = norm_layer
        self.drop_path_rate = drop_path_rate
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, self.num_register_tokens, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                DINOv3Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    rope_theta=self.rope_theta,
                    with_cp=with_cp,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

    @property
    def num_prefix_tokens(self):
        return 1 + self.num_register_tokens

    def get_prefix_tokens(self, batch_size):
        cls = self.cls_token.expand(batch_size, -1, -1)
        regs = self.register_tokens.expand(batch_size, -1, -1)
        return torch.cat([cls, regs], dim=1)
