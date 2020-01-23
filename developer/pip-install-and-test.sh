#!/bin/bash

if [[ ! $(basename $(pwd))='developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit
fi

conda create --force -y -n bornagain_test
conda activate bornagain_test
cd ..
pip install -e .
cd test
pytest
