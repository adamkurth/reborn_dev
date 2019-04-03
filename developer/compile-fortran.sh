#!/bin/bash

cd ..
rootdir=$(pwd)
#cd $rootdir/bornagain/analysis
#python -m numpy.f2py --quiet -c peaks.f90 -m peaks_f
cd $rootdir/bornagain/target
python -m numpy.f2py -c density.f90 -m density_f
#f2py --quiet -c peaks.f90 -m peaks_f
