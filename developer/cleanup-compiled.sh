#!/bin/bash

if [[ ! $(basename $(pwd))='developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit
fi

echo 'Cleaning compiled objects'
find .. -name '*.so' -type f -exec rm {} \;
find .. -name '*.dSYM' -type d -exec rm -r {} \;
find .. -name '*.pyc' -type f -exec rm {} \;