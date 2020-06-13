#!/bin/bash

if [[ ! $(basename "$(pwd)") = 'developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit
fi

find .. -name '*.so' -type f -exec rm {} \; &> /dev/null
find .. -name '*.dSYM' -type d -exec rm -r {} \; &> /dev/null
find .. -name '*.pyc' -type f -exec rm {} \; &> /dev/null
find .. -name 'reborn.egg-info' -type d -exec rm -r {} \; &> /dev/null
find .. -name build -type d -exec rm -r {} \; &> /dev/null
