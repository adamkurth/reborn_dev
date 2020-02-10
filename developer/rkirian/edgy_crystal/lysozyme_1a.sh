#!/bin/bash

export PYTHONPATH=${PYTHONPATH}:../../..

python edgy_crystal_3d.py --pdb_file 4ET8 \
                          --resolution 2e-10 \
                          --oversampling 4 \
                          --n_crystals 1 \
                          --crystal_length 1,1 \
                          --crystal_width 1,1 \
                          --gaussian_disorder_sigmas 0.0,0.0,0.0 \
                          --photon_energy_ev 8000 \
                          --view_density \
                          --view_intensities \
#                          --direct_molecular_transform
