#!/bin/bash

python -m numpy.f2py -c interpolations.f90 -m interpolations_f --quiet -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
python -m numpy.f2py -c wtf.f90 -m wtf_f --quiet -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION