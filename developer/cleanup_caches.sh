#!/bin/bash

if [[ ! $(basename "$(pwd)") = 'developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit
fi

find .. \( -path ../miniconda \) -prune -o -name '__pycache__' -type d -exec rm -r {} \; &> /dev/null
find .. \( -path ../miniconda \) -prune -o -name '.cache' -type d -exec rm -r {} \; &> /dev/null
find .. \( -path ../miniconda \) -prune -o -name '.pytest_cache' -type d -exec rm -r {} \; &> /dev/null
