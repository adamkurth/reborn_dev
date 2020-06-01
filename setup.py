from __future__ import division, absolute_import, print_function
import os
import shutil
from setuptools import find_packages
from numpy.distutils.core import setup, Extension
import datetime


with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["future", "numpy", "scipy", "h5py", "numba", "matplotlib", "pyqtgraph", "pyopencl"]

ext_modules = list()


#################################################################################################
# Fortran code
#################################################################################################

f2py_macros = [('NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION', '')]
extra_args = {}  # {'extra_compile_args': ['-static']}
omp_args = {}  # {'libraries': ['gomp'], 'extra_compile_args': ['-fopenmp']}
ext_modules.append(Extension(
      name='reborn.fortran.interpolations_f',
      sources=['reborn/fortran/interpolations.f90'],
      define_macros=f2py_macros,
      **extra_args
      ))
ext_modules.append(Extension(
      name='reborn.fortran.peaks_f',
      sources=['reborn/fortran/peaks.f90'],
      define_macros=f2py_macros,
      **extra_args,
      **omp_args
      ))
ext_modules.append(Extension(
      name='reborn.fortran.wtf_f',
      sources=['reborn/fortran/wtf.f90'],
      define_macros=f2py_macros,
      **extra_args
      ))
ext_modules.append(Extension(
      name='reborn.fortran.density_f',
      sources=['reborn/fortran/density.f90'],
      define_macros=f2py_macros,
      **extra_args
      ))

setup(name='reborn',
      version=datetime.date.today().strftime('%Y.%m.%d'),
      description='Diffraction analysis and simulation utilities',
      author='Richard A. Kirian',
      author_email='rkirian@asu.edu',
      url='https://rkirian.gitlab.io/reborn',
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
