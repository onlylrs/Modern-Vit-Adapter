import sys
import importlib.util
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

module_path = Path(__file__).resolve().parents[1] / 'mmseg_custom/models/backbones/kaa_modules.py'
spec = importlib.util.spec_from_file_location('kaa_modules', module_path)
kaa_modules = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kaa_modules)
KernelAlignedAdapter = kaa_modules.KernelAlignedAdapter
sampled_kernel_alignment_loss = kaa_modules.sampled_kernel_alignment_loss


def _run_all():
    test_kernel_aligned_adapter_preserves_token_shape_and_starts_as_residual()
    test_sampled_kernel_alignment_loss_ignores_void_labels_and_is_finite()


def test_kernel_aligned_adapter_preserves_token_shape_and_starts_as_residual():
    module = KernelAlignedAdapter(dim=8, cdim=6, rank=4, gate_init=0.0)
    x = torch.randn(2, 5, 8)
    c = torch.randn(2, 5, 6)

    out = module(x, c)

    assert out.shape == x.shape
    assert torch.allclose(out, x, atol=1e-6)


def test_sampled_kernel_alignment_loss_ignores_void_labels_and_is_finite():
    u = torch.rand(1, 4, 3) + 0.1
    v = torch.rand(1, 4, 3) + 0.1
    labels = torch.tensor([[[0, 0], [1, 255]]])

    loss = sampled_kernel_alignment_loss(u, v, labels, sample_size=4, ignore_index=255)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss >= 0


if __name__ == '__main__':
    _run_all()
