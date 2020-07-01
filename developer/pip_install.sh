#!/bin/bash

if [[ ! $(basename "$(pwd)") = 'developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit
fi

export NPY_DISTUTILS_APPEND_FLAGS=1
export NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

cd ..
pip -v install --no-index --no-deps --editable .  # possibly also the --user and --no-index flags
