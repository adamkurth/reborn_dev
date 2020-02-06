#!/bin/bash

if [[ ! $(basename "$(pwd)") = 'developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit
fi

./cleanup-everything.sh
./compile-fortran.sh
cd ../test
pytest
