#!/bin/bash

# Test the base of bornagain, with minimal dependencies.
conda create --no-default-packages --name bamin36 python=3.6 h5py scipy
echo activating...
source activate bamin36

echo testing...
echo $(which python)
cd ../test
python test_minimal_dependencies.py
