#!/bin/bash

if [[ "$(conda env list | grep reborn_minimal)" == "" ]]; then
  conda create --name reborn_minimal python=3 pytest scipy h5py
fi

source activate reborn_minimal
cd ..
pytest test/test_minimal_dependencies.py