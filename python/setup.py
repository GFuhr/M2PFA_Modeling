from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Common compiler and linker flags - combining all flags from pyxbld files
common_compile_args = [
    '-O2',
    '-march=native',
    '-fopenmp',
]
common_link_args = ['-fopenmp']

# Define extensions
extensions = [
    Extension(
        "advdiff.bicgstab",
        ["advdiff/bicgstab.pyx"],
        include_dirs=[np.get_include(), 'advdiff'],
        extra_compile_args=common_compile_args,
        extra_link_args=common_link_args,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
    Extension(
        "advdiff.operators_1d",
        ["advdiff/operators_1d.pyx"],
        include_dirs=[np.get_include(), 'advdiff'],
        depends=['advdiff/bicgstab.pxd'],  # Added because operators_1d depends on bicgstab.pxd
        extra_compile_args=common_compile_args,
        extra_link_args=common_link_args,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    )
]

setup(
    name="advdiff",
    version="0.1",
    packages=['advdiff'],
    package_dir={'advdiff': 'advdiff'},
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': 3,
            'boundscheck': False,
            'wraparound': False,
            'initializedcheck': False,
            'nonecheck': False,
            'cdivision': True,
        }
    ),
    install_requires=[
        'numpy',
        'cython',
    ],
) 