#!/bin/bash

if [[ ! $(basename "$(pwd)") = 'developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit
fi

find .. -name '__pycache__' -type d -exec rm -r {} \; &> /dev/null
find .. -name '.cache' -type d -exec rm -r {} \; &> /dev/null
find .. -name '.pytest_cache' -type d -exec rm -r {} \; &> /dev/null
