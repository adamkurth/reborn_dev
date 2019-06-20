#!/bin/bash

cd ..
rootdir=$(pwd)
cd $rootdir/bornagain/target
python -m numpy.f2py --quiet -c density.f90 -m density_f --quiet -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
cd $rootdir/bornagain/analysis
python -m numpy.f2py --quiet -c peaks.f90 -m peaks_f --quiet -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

