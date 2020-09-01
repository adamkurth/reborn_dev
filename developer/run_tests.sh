#!/bin/bash

if [[ ! $(basename "$(pwd)") = 'developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit
fi

export PYTHONPATH=$(cd ..; pwd):$PYTHONPATH
cd ../test
pytest $@
