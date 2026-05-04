# Modern ViT-Adapter

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/onlylrs/Modern-Vit-Adapter/blob/main/LICENSE) [![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://github.com/onlylrs/Modern-Vit-Adapter/blob/main/INSTALL.md) [![PyTorch](https://img.shields.io/badge/PyTorch-2.9%20%28CUDA%2012.8%29-ee4c2c.svg)](https://pytorch.org/) [![Release](https://img.shields.io/github/v/release/onlylrs/Modern-Vit-Adapter?label=release)](https://github.com/onlylrs/Modern-Vit-Adapter/releases)

Built on the [ViT-Adapter](https://arxiv.org/abs/2205.08534) architecture, this project modernizes the original codebase for PyTorch 2.x+ with easy installation. 

The versions mmcv==1.4.2, mmdet==2.22.0, and mmseg==0.20.2 are kept, with compatibility updates for the latest PyTorch. See the log of these fixes [here](third_party/README.md).

**News🔥**

`2026-05-03` Added support for [ViT-CoMer](https://github.com/Traffic-X/ViT-CoMer) and [ViT-Split](https://github.com/JackYFL/ViT-Split).

`2026-04-30` Added support for [MaskDINO](https://github.com/IDEA-Research/MaskDINO) and two adapters for [DINOv3](https://github.com/facebookresearch/dinov3). See [Release v1.1.0](https://github.com/onlylrs/Modern-Vit-Adapter/releases/tag/v1.1.0).

# Installation
See [INSTALL.md](INSTALL.md). 

# Train and test
**Important:** To activate the environment, you must perform *both* of the following steps **every time**:

1. Use your environment manager to activate the environment (for example: `micromamba activate torch29`).
2. Run `source env.sh` to link to mmcv, mmdet and mmseg.

For training and testing, the usage is exactly the same with the original ViT-Adapter project, see [detection/README.md](detection/README.md) and [segmentation/README.md](segmentation/README.md). 
Apart from the original usage, we also support the DINOv3 backbones. Please refer to [scripts/example.train.sh](scripts/example.train.sh) and [scripts/example.test.sh](scripts/example.test.sh).

# Todo
- [x] Support [ViT-CoMer](https://github.com/Traffic-X/ViT-CoMer) - another excellent adapter designed for ViT-based backbones
- [x] Support [MaskDINO](https://github.com/IDEA-Research/MaskDINO) - a solid task head for detection and segmentation. 

