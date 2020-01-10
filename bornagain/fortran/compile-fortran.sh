#!/bin/bash

f2py="python -m numpy.f2py"
flags="-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"

${f2py} -c interpolations.f90 -m interpolations_f ${flags}
${f2py} -c wtf.f90 -m wtf_f ${flags}
${f2py} -c peaks.f90 -m peaks_f ${flags} --f90flags='-fopenmp -O2' -lgomp # -static
