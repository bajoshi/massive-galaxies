from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("model_mods_cython.pyx"),
    include_dirs=[numpy.get_include()]
)