from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = "0.0.1"

# Define the C++ extension modules
ext_modules = [
    CUDAExtension('gpu_kernels', [
        './relu_binding.cpp',
        './relu.cu',
    ])
]

setup(
    name="custom_kernels",
    version=__version__,
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension}
)