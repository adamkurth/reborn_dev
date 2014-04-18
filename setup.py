from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
import os

os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"

extensions = Extension("pydiffract.simulate.cycore",
                       ["pydiffract/simulate/cycore.pyx"],
                       include_dirs=[numpy.get_include()],
                       extra_compile_args=['-fopenmp'],
                       extra_link_args=['-fopenmp'])

setup(name='pydiffract',
      version='0.0.1',
      author="Richard A. Kirian",
      author_email="rkirian@gmail.com",
      description='Diffraction analysis tools',
      packages=["pydiffract", "pydiffract.viewers", "pydiffract.simulate"],
      package_dir={"pydiffract": "pydiffract"},
      ext_modules=cythonize(extensions))
