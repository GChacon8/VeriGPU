"""
setup.py — Build script for VeriGPU PyTorch backend extension.

Conditionally links against libverigpu_runtime.so if available.
If the runtime is built (build/runtime-linux/libverigpu_runtime.so exists),
the extension can use VERIGPU_USE_HW=1 for hardware mode.
If not, the extension works in host-CPU-only mode.

Usage (from the pytorch_backend/ directory, with venv activated):
    pip install -e . --no-build-isolation
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
runtime_dir = os.path.join(repo_root, 'build', 'runtime-linux')
runtime_lib = os.path.join(runtime_dir, 'libverigpu_runtime.so')

hw_available = os.path.exists(runtime_lib)

extra_compile_args = ['-std=c++17', '-fno-strict-aliasing']
extra_link_args = []
libraries = []
library_dirs = []

if hw_available:
    print(f"[setup.py] Found libverigpu_runtime.so at {runtime_dir}")
    print(f"[setup.py] Enabling VERIGPU_HW_AVAILABLE")
    extra_compile_args.append('-DVERIGPU_HW_AVAILABLE')
    library_dirs.append(runtime_dir)
    libraries.append('verigpu_runtime')
    # rpath so the .so is found at runtime without LD_LIBRARY_PATH
    extra_link_args.append(f'-Wl,-rpath,{runtime_dir}')
else:
    print(f"[setup.py] libverigpu_runtime.so NOT found at {runtime_dir}")
    print(f"[setup.py] Building in host-only mode (no hardware simulation)")

setup(
    name='verigpu_backend',
    version='0.2.0',
    description='VeriGPU custom backend for PyTorch (with optional HW simulation)',
    ext_modules=[
        CppExtension(
            name='_verigpu_C',
            sources=['verigpu_backend.cpp'],
            extra_compile_args=extra_compile_args,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_link_args=extra_link_args,
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    },
)
