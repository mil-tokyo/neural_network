from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension(name = "convolve", sources=["convolve.pyx"], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp']),
    Extension(name = 'pool', sources=["pool.pyx"], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp']),
]

setup(
    name = "convolve function",
    cmdclass = {"build_ext" : build_ext},
    ext_modules = ext_modules,
)
