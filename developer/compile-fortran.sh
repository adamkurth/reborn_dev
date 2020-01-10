#!/bin/bash

cd ..
rootdir=$(pwd)

echo root directory: ${rootdir}

#cd ${rootdir}/bornagain/target
#python -m numpy.f2py -c density.f90 -m density_f --quiet -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

cd ${rootdir}/bornagain/fortran
source compile-fortran.sh
