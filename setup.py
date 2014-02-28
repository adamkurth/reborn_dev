from distutils.core import setup, Extension
from Cython.Build import cythonize

extensions = Extension("pydiffract.simulate.core", ["pydiffract/simulate/core.pyx"])

setup(name='pydiffract',
      version='0.0.1',
      author="Richard A. Kirian",
      author_email="rkirian@gmail.com",
      description='Diffraction analysis tools',
      packages=["pydiffract", "pydiffract.viewers", "pydiffract.simulate"],
      package_dir={"pydiffract": "pydiffract"},
      ext_modules=cythonize(extensions))