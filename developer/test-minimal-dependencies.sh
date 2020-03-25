#!/bin/bash

if [[ ! $(basename "$(pwd)") = 'developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit
fi

# Test the base of reborn, with minimal dependencies.
conda create --no-default-packages --name bamin36 python=3.8 h5py scipy
echo activating...
source activate bamin36

echo testing...
command -v python
cd ../test || return
python test_minimal_dependencies.py
