#!/bin/bash

cd ../bornagain/analysis
f2py --quiet -c peaks.f90 -m peaks_f
