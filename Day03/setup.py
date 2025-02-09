from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = "0.0.5"

# Define the C++ extension modules
ext_modules = [
    CUDAExtension('gpu_kernels', [
        './blur_binding.cpp',
        './blur.cu',
    ])
]

setup(
    name="cuda_basics",
    version=__version__,
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension}
)