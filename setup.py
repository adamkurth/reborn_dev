from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = Extension("pydiffract.simulate.core", ["pydiffract/simulate/core.pyx"], include_dirs=[numpy.get_include()])

setup(name='pydiffract',
      version='0.0.1',
      author="Richard A. Kirian",
      author_email="rkirian@gmail.com",
      description='Diffraction analysis tools',
      packages=["pydiffract", "pydiffract.viewers", "pydiffract.simulate"],
      package_dir={"pydiffract": "pydiffract"},
      ext_modules=cythonize(extensions))
