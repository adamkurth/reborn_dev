#!/bin/bash

if [[ ! $(basename "$(pwd)") = 'developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit
fi

cd ../examples/simulations || return

python density.py noplots
python simulate_lattice.py noplots
python simulate_pdb.py noplots