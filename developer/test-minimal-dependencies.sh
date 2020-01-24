#!/bin/bash

if [[ ! $(basename $(pwd))='developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit
fi

# Test the base of bornagain, with minimal dependencies.
conda create --no-default-packages --name bamin36 python=3.8 h5py scipy
echo activating...
source activate bamin36

echo testing...
echo $(which python)
cd ../test
python test_minimal_dependencies.py
