"""
setup.py — Build script for VeriGPU PyTorch backend extension.

Usage (from the pytorch_backend/ directory, with venv activated):
    pip install -e .

This compiles verigpu_backend.cpp into a shared library (_verigpu_C)
that Python can import. The '-e' flag makes it editable: if you change
the .cpp file, just run 'pip install -e .' again to recompile.
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='verigpu_backend',
    version='0.1.0',
    description='VeriGPU custom backend for PyTorch',
    ext_modules=[
        CppExtension(
            name='_verigpu_C',
            sources=['verigpu_backend.cpp'],
            extra_compile_args=['-std=c++17'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    },
)
