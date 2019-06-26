#!/bin/bash

cd ..
rootdir=$(pwd)

echo root directory: $rootdir

cd $rootdir/bornagain/target
python -m numpy.f2py -c density.f90 -m density_f --quiet -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

cd $rootdir/bornagain/analysis
#python -m numpy.f2py -c peaks.f90 -m peaks_f --quiet -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# Use this below if want openmp.
python -m numpy.f2py -c --f90flags='-fopenmp -O2' peaks.f90 -m peaks_f --quiet -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

cd $rootdir/bornagain/fortran
./compile-fortran.sh
