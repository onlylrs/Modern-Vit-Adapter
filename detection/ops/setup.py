# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

import glob
import os
import sys

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

requirements = ['torch', 'torchvision']


def get_extra_cuda_paths():
    paths = {'include_dirs': [], 'library_dirs': []}

    cuda_home = os.getenv('CUDA_HOME') or os.getenv('CUDA_PATH') or ''
    candidates = []
    if cuda_home:
        candidates.extend([
            os.path.join(cuda_home, 'targets', 'x86_64-linux', 'include'),
            os.path.join(cuda_home, 'targets', 'x86_64-linux', 'lib'),
            os.path.join(cuda_home, 'include'),
            os.path.join(cuda_home, 'lib64'),
            os.path.join(cuda_home, 'lib'),
        ])

    site_packages = os.path.join(
        sys.prefix, 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}',
        'site-packages')
    candidates.extend(glob.glob(os.path.join(site_packages, 'nvidia', '*', 'include')))
    candidates.extend(glob.glob(os.path.join(site_packages, 'nvidia', '*', 'lib')))
    candidates.extend(glob.glob(os.path.join(site_packages, 'nvidia', '*', 'lib64')))

    for path in candidates:
        if not os.path.isdir(path):
            continue
        if path.endswith('include'):
            paths['include_dirs'].append(path)
        else:
            paths['library_dirs'].append(path)

    paths['include_dirs'] = list(dict.fromkeys(paths['include_dirs']))
    paths['library_dirs'] = list(dict.fromkeys(paths['library_dirs']))
    return paths


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, 'src')

    main_file = glob.glob(os.path.join(extensions_dir, '*.cpp'))
    source_cpu = glob.glob(os.path.join(extensions_dir, 'cpu', '*.cpp'))
    source_cuda = glob.glob(os.path.join(extensions_dir, 'cuda', '*.cu'))

    sources = main_file + source_cpu
    extension = CppExtension
    extra_compile_args = {'cxx': []}
    define_macros = []
    extra_cuda_paths = get_extra_cuda_paths()
    library_dirs = []

    if os.name != 'nt':
        extra_compile_args['cxx'] = ['-std=c++17']

    if (torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1') and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [('WITH_CUDA', None)]
        extra_compile_args['nvcc'] = [
            '-DCUDA_HAS_FP16=1',
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
            '-std=c++17',
        ]
    else:
        raise NotImplementedError('Cuda is not availabel')

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir] + extra_cuda_paths['include_dirs']
    ext_modules = [
        extension(
            'MultiScaleDeformableAttention',
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            library_dirs=library_dirs + extra_cuda_paths['library_dirs'],
        )
    ]
    return ext_modules


setup(
    name='MultiScaleDeformableAttention',
    version='1.0',
    author='Weijie Su',
    url='https://github.com/fundamentalvision/Deformable-DETR',
    description=
    'PyTorch Wrapper for CUDA Functions of Multi-Scale Deformable Attention',
    packages=find_packages(exclude=(
        'configs',
        'tests',
    )),
    ext_modules=get_extensions(),
    cmdclass={'build_ext': torch.utils.cpp_extension.BuildExtension},
)
