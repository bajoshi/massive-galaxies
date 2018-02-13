from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("model_mods_cython.pyx"),
    include_dirs=[numpy.get_include()]
)   