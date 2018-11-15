#!/bin/bash

cd ../bornagain/analysis
f2py -c peaks.f90 -m peaks_f
