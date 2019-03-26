#!/usr/bin/env bash

cd ../examples

cd simulations
python density.py noplots
python simulate_pdb.py noplots
python water_background.py noplots
