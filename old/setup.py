from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
import os

os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"

extensions = Extension("bornagain.simulate.cycore",
                       ["bornagain/simulate/cycore.pyx"],
                       include_dirs=[numpy.get_include()],
                       extra_compile_args=['-fopenmp'],
                       extra_link_args=['-fopenmp'])

setup(name='bornagain',
      version='0.0.1',
      author="Richard A. Kirian",
      author_email="rkirian@gmail.com",
      description='Diffraction analysis tools',
      packages=["bornagain", "bornagain.viewers", "bornagain.simulate"],
      package_dir={"bornagain": "bornagain"},
      ext_modules=cythonize(extensions))
