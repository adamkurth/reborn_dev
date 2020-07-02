#!/bin/bash

if [[ ! $(basename "$(pwd)") = 'developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit
fi

cd ..
pip -v install --no-index --user --no-deps --editable .
