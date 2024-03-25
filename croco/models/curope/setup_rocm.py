# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# compile for all possible ROCm architectures
all_rocm_archs = ["gfx1100"] # only for gfx1100
# "gfx803", "gfx900", "gfx906", "gfx908", "gfx1010", "gfx1011", "gfx1012", "gfx1030" optional

setup(
    name = 'curope',
    ext_modules = [
        CUDAExtension(
                name='curope',
                sources=[
                    "curope.cpp",
                    "kernels.cu",
                ],
                extra_compile_args = dict(
                    nvcc=['-O3', '--offload-arch=' + ','.join(all_rocm_archs)],
                    cxx=['-O3'])
                )
    ],
    cmdclass = {
        'build_ext': BuildExtension
    })
