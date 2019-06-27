#!/usr/bin/env bash

cd ../examples/simulations

python density.py noplots
python simulate_lattice.py noplots
python simulate_pdb.py noplots
python simulate_two_atoms.py noplots
python spheres.py noplots
python water_background.py noplots
