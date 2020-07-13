#!/bin/bash

if [[ ! $(basename "$(pwd)") = 'developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit 1
fi

cd ../reborn/fortran || exit

export NPY_DISTUTILS_APPEND_FLAGS=1
export NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

f2py="python -m numpy.f2py"
flags="" #-Wno-unused-function" #-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION -Wno-unused-function"

${f2py} -c utils.f90 -m utils_f ${flags}
${f2py} -c interpolations.f90 -m interpolations_f ${flags}
${f2py} -c fortran_indexing.f90 -m fortran_indexing_f ${flags}
${f2py} -c density.f90 -m density_f ${flags}
${f2py} -c peaks.f90 -m peaks_f ${flags} --f90flags='-fopenmp -O2' -lgomp
