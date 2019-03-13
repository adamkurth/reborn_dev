#!/bin/bash

cd ../bornagain/analysis
python -m numpy.f2py --quiet -c peaks.f90 -m peaks_f
#f2py --quiet -c peaks.f90 -m peaks_f
