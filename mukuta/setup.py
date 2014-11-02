from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
        cmdclass={'build_ext': build_ext},
        ext_modules=[Extension("convolve3d",["convolve3d.pyx"],extra_compile_args=['-fopenmp'],extra_link_args=['-fopenmp']),Extension("convolveback",["convolveback.pyx"],extra_compile_args=['-fopenmp'],extra_link_args=['-fopenmp'])]
        )
