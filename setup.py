import os
from setuptools import find_packages
from numpy.distutils.core import setup, Extension
import datetime


with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["scipy", "h5py"]

ext_modules = list()


#################################################################################################
# Fortran code
#################################################################################################
os.environ['NPY_DISTUTILS_APPEND_FLAGS'] = '1'
os.environ['NPY_NO_DEPRECATED_API'] = 'NPY_1_7_API_VERSION'
f2py_macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'), ('NPY_DISTUTILS_APPEND_FLAGS', '1')]
#extra_args = {'extra_compile_args': ['-Wno-unused-function', '-fopenmp']}
#omp_args = {'extra_compile_args': ['-fopenmp']}  # {'libraries': ['gomp'], 'extra_compile_args': ['-fopenmp']}
ext_modules.append(Extension(
      name='reborn.fortran.utils_f',
      sources=['reborn/fortran/utils.f90'],
      define_macros=f2py_macros,
      extra_compile_args=['-Wno-unused-function']
      ))
ext_modules.append(Extension(
      name='reborn.fortran.interpolations_f',
      sources=['reborn/fortran/interpolations.f90'],
      define_macros=f2py_macros,
      extra_compile_args=['-Wno-unused-function']
      ))
ext_modules.append(Extension(
      name='reborn.fortran.peaks_f',
      sources=['reborn/fortran/peaks.f90'],
      define_macros=f2py_macros,
      extra_compile_args=['-Wno-unused-function', '-fopenmp', '-lgomp']
      ))
ext_modules.append(Extension(
      name='reborn.fortran.omp_test_f',
      sources=['reborn/fortran/omp_test.f90'],
      define_macros=f2py_macros,
      extra_compile_args=['-Wno-unused-function', '-fopenmp', '-lgomp']
      ))
ext_modules.append(Extension(
      name='reborn.fortran.fortran_indexing_f',
      sources=['reborn/fortran/fortran_indexing.f90'],
      define_macros=f2py_macros,
      extra_compile_args=['-Wno-unused-function']
      ))
ext_modules.append(Extension(
      name='reborn.fortran.density_f',
      sources=['reborn/fortran/density.f90'],
      define_macros=f2py_macros,
      extra_compile_args=['-Wno-unused-function']
      ))

setup(name='reborn',
      version=datetime.date.today().strftime('%Y.%m.%d').replace('.0', '.'),
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
