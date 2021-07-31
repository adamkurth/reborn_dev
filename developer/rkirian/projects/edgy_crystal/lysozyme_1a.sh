#!/bin/bash

export PYTHONPATH=${PYTHONPATH}:../../..

python edgy_crystal_3d.py --pdb_file 1jb0 \
                          --resolution 10e-10 \
                          --oversampling 1 \
                          --n_crystals 1 \
                          --crystal_length 1,1 \
                          --crystal_width 1,1 \
                          --gaussian_disorder_sigmas 0.0,0.0,0.0 \
                          --photon_energy_ev 8000 \
                          --view_crystal \
                          --view_density \
#                          --view_intensities \
#                          --direct_molecular_transform
