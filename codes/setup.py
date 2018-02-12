from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'model modifications in cython',
  ext_modules = cythonize("model_mods_cython.pyx"),
)