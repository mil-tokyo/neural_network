from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
name = 'Convolutional Neural Network',
ext_modules = cythonize("convolve3d.pyx"),
include_dirs=[numpy.get_include()]
)

