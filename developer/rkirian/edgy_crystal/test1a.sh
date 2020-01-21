#!/bin/bash

cd $(dirname $0)

export PYTHONPATH=${PYTHONPATH}:../../..

python edgy_crystal_3d.py --pdb_file 1jb0 \
                          --resolution 10e-10 \
                          --oversampling 4 \
                          --n_crystals 10 \
                          --crystal_length 6,10 \
                          --crystal_width 2,4 \
                          --gaussian_disorder_sigmas 0.05,0.05,0.05 \
                          --photon_energy_ev 8000 \
                          --view_crystal \
                          --view_density \
                          --view_intensities \
                          --save_results \
                          --run_number 1
