# Installation Guide
Any env manager works (e.g. conda, miniconda, mamba, micromamba, uv). [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) is shown as the example.

Create environment:
```bash
micromamba create -n torch29 python=3.12
micromamba activate torch29
```
Install PyTorch and cuda nvcc (example: CUDA12.8 & torch2.9.1):
```bash
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128

micromamba install -c nvidia -c conda-forge cuda-nvcc=12.8
```
Install relevant packages:
```bash
pip install -U pip
pip install "setuptools<82" wheel packaging ninja Cython
pip install addict pyyaml pillow yapf opencv-python-headless matplotlib six terminaltables timm numpy
```

Compile mmcv:
```bash
export CUDA_HOME="$HOME/micromamba/envs/torch29"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

cd third_party/openmmlab/mmcv
python setup.py build_ext --inplace

```

Above, let's set `TORCH_CUDA_ARCH_LIST` to match your GPU version(s). The example

```
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
```

covers most modern GPUs (A100, RTX 30xx, Ada, H100). Pick only those you need or just leave all for broader compatibility.

Test mmcv:
```bash
python - <<'PY'
import sys
sys.path.insert(0, ".")
import mmcv
print("mmcv:", mmcv.__version__)
from mmcv import ops
print("mmcv.ops imported")
PY
```

Test mmdet and mmseg:
```bash
source env.sh
python - <<'PY'
import mmdet
import mmseg
print("mmdet:", mmdet.__version__)
print("mmseg:", mmseg.__version__)
PY
```

Compile MultiScaleDeformableAttention ops:
```bash
source env.sh
cd detection/ops
python setup.py build_ext --inplace
```
Test ops:
```bash
source env.sh

python - <<'PY'
import sys
import torch
sys.path.insert(0, "detection/ops")
import MultiScaleDeformableAttention as MSDA
print("torch:", torch.__version__)
print("module:", MSDA.__file__)
print("has forward:", hasattr(MSDA, "ms_deform_attn_forward"))
print("has backward:", hasattr(MSDA, "ms_deform_attn_backward"))
PY

python - <<'PY'
import torch
from detection.ops.modules import MSDeformAttn
print("MSDeformAttn import ok:", MSDeformAttn)
PY
```

The installation should be smooth. Feel free to post an issue if encoutering any problems.
