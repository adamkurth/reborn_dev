#!/bin/bash

if [[ ! $(basename "$(pwd)") = 'developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit
fi
export NPY_DISTUTILS_APPEND_FLAGS=1
cd ..
pip -v install --no-deps --editable .  # possibly also the --user and --no-index flags
