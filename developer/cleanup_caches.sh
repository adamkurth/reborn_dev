#!/bin/bash

if [[ ! $(basename "$(pwd)") = 'developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit
fi

echo 'Cleaning caches'
find .. -name '__pycache__' -type d -exec rm -r {} \;
find .. -name '.cache' -type d -exec rm -r {} \;
find .. -name '*.pyc' -type f -exec rm {} \;
find .. -name '.pytest_cache' -type d -exec rm -r {} \;
