What we have changed ...

# Compatibility notes

The project does not rely on reinstalling newer OpenMMLab releases. Instead, the original lightweight stack is kept in `third_party/openmmlab/` and patched to work with the current PyTorch/CUDA toolchain.

Main changes to `mmcv`:

- Build system updated for modern PyTorch extensions:
  - use `C++17` instead of the old `C++14` default
  - discover CUDA headers/libs from both `CUDA_HOME` and the current env's `site-packages/nvidia/...`
  - allow `FORCE_CUDA=1` for building on login nodes without visible GPUs
- CUDA/C++ API migrations for PyTorch 2.x:
  - remove deprecated `THC` header usage
  - replace old tensor accessors such as `.data<T>()` / `.type()` with `data_ptr<T>()`, `scalar_type()`, and `is_cuda()`
  - replace `AT_ERROR` / `AT_ASSERTM` with `TORCH_CHECK`
- Runtime compatibility fixes:
  - adapt `MMDistributedDataParallel` to newer PyTorch DDP internals
  - fix MMCV scatter helpers for newer `_get_stream` device handling
  - patch `yapf` and `numpy` incompatibilities in old MMCV utilities

How `mmdet` and `mmseg` are used:

- `mmdet` and `mmseg` are not separately installed from pip wheels.
- They are imported directly from the vendored source tree through `env.sh`.
- This keeps all local compatibility patches active and avoids mixing patched code with external site-packages copies.

Main changes to project ops in `detection/ops`:

- `MultiScaleDeformableAttention` is a local CUDA/C++ extension and must be compiled.
- The build script was updated to:
  - use `C++17`
  - discover CUDA include/lib directories from the active environment
  - support `FORCE_CUDA=1` when needed
- The CUDA sources were migrated away from removed PyTorch 1.x / THC APIs.
- The Python wrapper was updated for modern AMP (`torch.amp`) and now adds `detection/ops` to `sys.path` before importing the compiled module.
- The compiled shared object is expected to live inside `detection/ops/`, and the runtime loads it from there.

In short:

- `mmcv` needs to be compiled
- `mmdet` / `mmseg` are used from source via `PYTHONPATH`
- `detection/ops` needs to be compiled locally
- the rest of the project code was then adjusted to run on the modern stack