import copy
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def gen_encoder_output_proposals(memory: Tensor, memory_padding_mask: Tensor,
                                 spatial_shapes: Tensor):
    n, _, _ = memory.shape
    proposals = []
    cur = 0
    for lvl, (h, w) in enumerate(spatial_shapes):
        h_int, w_int = int(h.item()), int(w.item())
        mask_flatten = memory_padding_mask[:, cur:cur + h_int * w_int].view(
            n, h_int, w_int, 1)
        valid_h = torch.sum(~mask_flatten[:, :, 0, 0], 1)
        valid_w = torch.sum(~mask_flatten[:, 0, :, 0], 1)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, h_int - 1, h_int, dtype=torch.float32,
                           device=memory.device),
            torch.linspace(0, w_int - 1, w_int, dtype=torch.float32,
                           device=memory.device),
        )
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
        scale = torch.cat([valid_w.unsqueeze(-1), valid_h.unsqueeze(-1)], 1)
        scale = scale.view(n, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(n, -1, -1, -1) + 0.5) / scale
        wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
        proposals.append(torch.cat((grid, wh), -1).view(n, -1, 4))
        cur += h_int * w_int

    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = ((output_proposals > 0.01) &
                              (output_proposals < 0.99)).all(-1, keepdim=True)
    output_proposals = torch.log(output_proposals / (1 - output_proposals))
    output_proposals = output_proposals.masked_fill(
        memory_padding_mask.unsqueeze(-1), float('inf'))
    output_proposals = output_proposals.masked_fill(
        ~output_proposals_valid, float('inf'))

    output_memory = memory
    output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), 0)
    output_memory = output_memory.masked_fill(~output_proposals_valid, 0)
    return output_memory, output_proposals


def gen_sineembed_for_position(pos_tensor):
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(),
                         pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(),
                         pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        return torch.cat((pos_y, pos_x), dim=2)
    if pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        h_embed = pos_tensor[:, :, 3] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_h = h_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(),
                             pos_w[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(),
                             pos_h[:, :, 1::2].cos()), dim=3).flatten(2)
        return torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    raise ValueError(f'Unknown pos tensor last dim: {pos_tensor.size(-1)}')


def _get_activation_fn(activation):
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    if activation == 'prelu':
        return nn.PReLU()
    if activation == 'selu':
        return F.selu
    raise RuntimeError(f'activation should be relu/gelu, not {activation}.')


def _get_clones(module, n, layer_share=False):
    if layer_share:
        return nn.ModuleList([module for _ in range(n)])
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])
