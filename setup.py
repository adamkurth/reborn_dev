from __future__ import division, absolute_import, print_function

from setuptools import find_packages
from numpy.distutils.core import setup, Extension

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["future", "numpy", "scipy", "numba", "matplotlib", "h5py", "pyqtgraph", "pyopengl", "pyopencl"]

ext_modules = list()


#################################################################################################
# Fortran code
#################################################################################################

f2py_macros = [('NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION', '')]
ext_modules.append(Extension(
      name='bornagain.fortran.interpolations_f',
      sources=['bornagain/fortran/interpolations.f90'],
      define_macros=f2py_macros
      ))
ext_modules.append(Extension(
      name='bornagain.fortran.peaks_f',
      sources=['bornagain/fortran/peaks.f90'],
      define_macros=f2py_macros
      # f2py_options=[],
      # define_macros=[('F2PY_REPORT_ON_ARRAY_COPY', '1')],
      # # this is the flag gfortran needs to process OpenMP directives
      # libraries=['gomp'],
      # extra_compile_args=['-fopenmp'],
      # extra_link_args=[],
      ))
ext_modules.append(Extension(
      name='bornagain.fortran.wtf_f',
      sources=['bornagain/fortran/wtf.f90'],
      define_macros=f2py_macros
      ))
ext_modules.append(Extension(
      name='bornagain.fortran.density_f',
      sources=['bornagain/fortran/density.f90'],
      define_macros=f2py_macros
      ))

setup(name='bornagain',
      version='0.1',
      description='Diffraction analysis and simulation utilities',
      # long_description=readme,
      # long_description_content_type="text/markdown",
      author='Richard A. Kirian',
      author_email='rkirian@asu.edu',
      url='https://rkirian.gitlab.io/bornagain',
      # package_dir={'bornagain': find_packages()},
      packages=find_packages(),
      ext_modules=ext_modules,
      install_requires=requirements,
      classifiers=[
            "Development Status :: 4 - Beta",
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
      ],
      zip_safe=False,
      )
