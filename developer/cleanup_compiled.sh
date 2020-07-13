#!/bin/bash

if [[ ! $(basename "$(pwd)") = 'developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit
fi

find .. \( -path ../miniconda \) -prune -o -name '*.so' -type f -exec rm {} \; &> /dev/null
find .. \( -path ../miniconda \) -prune -o -name '*.dSYM' -type d -exec rm -r {} \; &> /dev/null
find .. \( -path ../miniconda \) -prune -o -name '*.pyc' -type f -exec rm {} \; &> /dev/null
find .. \( -path ../miniconda \) -prune -o -name 'reborn.egg-info' -type d -exec rm -r {} \; &> /dev/null
find .. \( -path ../miniconda \) -prune -o -name build -type d -exec rm -r {} \; &> /dev/null
