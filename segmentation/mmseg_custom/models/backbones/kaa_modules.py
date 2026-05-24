import torch
import torch.nn as nn
import torch.nn.functional as F


class KernelAlignedAdapter(nn.Module):
    def __init__(self, dim, cdim=None, rank=64, gate_init=1e-3, eps=1e-6):
        super().__init__()
        cdim = cdim or dim
        self.eps = eps
        self.x_q = nn.Linear(dim, rank)
        self.x_k = nn.Linear(dim, rank)
        self.c_q = nn.Linear(cdim, rank)
        self.c_k = nn.Linear(cdim, rank)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.last_kernel_factors = None

    def positive(self, x):
        return F.softplus(x) + self.eps

    def forward(self, x, c, return_kernel=False):
        xs_q = self.positive(self.x_q(x))
        xs_k = self.positive(self.x_k(x))
        cg_q = self.positive(self.c_q(c))
        cg_k = self.positive(self.c_k(c))

        u = xs_q * cg_q
        v = xs_k * cg_k
        value = self.value(x)

        kv = torch.einsum('bnr,bnd->brd', v, value)
        z = torch.einsum('bnr,brd->bnd', u, kv)
        normalizer = torch.einsum('bnr,br->bn', u, v.sum(dim=1))
        z = z / (normalizer.unsqueeze(-1) + self.eps)
        out = x + self.gate * self.out(z)

        if return_kernel:
            self.last_kernel_factors = (u, v)
        else:
            self.last_kernel_factors = None
        return out


class AdditiveKernelAdapter(KernelAlignedAdapter):
    def forward(self, x, c, return_kernel=False):
        xs_q = self.positive(self.x_q(x))
        xs_k = self.positive(self.x_k(x))
        cg_q = self.positive(self.c_q(c))
        cg_k = self.positive(self.c_k(c))

        u = xs_q + cg_q
        v = xs_k + cg_k
        value = self.value(x)

        kv = torch.einsum('bnr,bnd->brd', v, value)
        z = torch.einsum('bnr,brd->bnd', u, kv)
        normalizer = torch.einsum('bnr,br->bn', u, v.sum(dim=1))
        z = z / (normalizer.unsqueeze(-1) + self.eps)
        out = x + self.gate * self.out(z)

        if return_kernel:
            self.last_kernel_factors = (u, v)
        else:
            self.last_kernel_factors = None
        return out


class FeatureFusionAdapter(nn.Module):
    def __init__(self, dim, cdim=None, hidden_ratio=0.25, gate_init=1e-3):
        super().__init__()
        cdim = cdim or dim
        hidden_dim = max(dim, int(dim * hidden_ratio))
        self.fuse = nn.Sequential(
            nn.LayerNorm(dim + cdim),
            nn.Linear(dim + cdim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.last_kernel_factors = None

    def forward(self, x, c, return_kernel=False):
        del return_kernel
        self.last_kernel_factors = None
        return x + self.gate * self.fuse(torch.cat([x, c], dim=-1))


def sampled_kernel_alignment_loss(u, v, labels, sample_size=1024, ignore_index=255, token_hw=None, eps=1e-6):
    if labels.dim() == 4:
        labels = labels.squeeze(1)
    labels = labels.long()
    if token_hw is None:
        side = int(u.shape[1] ** 0.5)
        token_hw = (side, side)
    labels = F.interpolate(labels.unsqueeze(1).float(), size=token_hw, mode='nearest')
    labels = labels.squeeze(1).long().flatten(1)

    bsz, num_tokens, _ = u.shape
    count = min(sample_size, num_tokens)
    if count <= 1:
        return u.sum() * 0.0

    losses = []
    for batch_idx in range(bsz):
        valid = torch.nonzero(labels[batch_idx] != ignore_index, as_tuple=False).flatten()
        if valid.numel() <= 1:
            continue
        if valid.numel() > count:
            perm = torch.randperm(valid.numel(), device=valid.device)[:count]
            valid = valid[perm]
        ub = F.normalize(u[batch_idx, valid], dim=-1)
        vb = F.normalize(v[batch_idx, valid], dim=-1)
        kernel = ub @ vb.t()
        target = (labels[batch_idx, valid, None] == labels[batch_idx, valid][None, :]).float()
        kernel = kernel.flatten()
        target = target.flatten()
        losses.append(1.0 - F.cosine_similarity(kernel, target, dim=0, eps=eps))
    if not losses:
        return u.sum() * 0.0
    return torch.stack(losses).mean()
