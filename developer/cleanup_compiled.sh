#!/bin/bash

if [[ ! $(basename "$(pwd)") = 'developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit
fi

echo cleaning compiled objects
find ../reborn/fortran -name '*.md5' -type f -exec rm {} \; &> /dev/null
find .. -name '*.so' -type f -exec rm {} \; &> /dev/null
find .. -name '*.dSYM' -type d -exec rm -r {} \; &> /dev/null
find .. -name '*.pyc' -type f -exec rm {} \; &> /dev/null
find .. -name 'reborn.egg-info' -type d -exec rm -r {} \; &> /dev/null
find .. -name build -type d -exec rm -r {} \; &> /dev/null
echo "done"