from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='occ_pool_ext',
    ext_modules=[
        CUDAExtension(
            name='occ_pool_ext',
            sources=[
                'src/occ_pool.cpp',
                'src/occ_pool_cuda.cu',
            ],
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': ['-O2'],
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
