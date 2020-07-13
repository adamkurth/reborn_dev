#!/bin/bash

if [[ ! $(basename "$(pwd)") = 'developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit
fi

export NPY_DISTUTILS_APPEND_FLAGS=1
export NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

cd ..
<<<<<<< HEAD
pip -v install --no-index --user --no-deps --editable .
=======
pip -v install --no-index --no-deps --editable .  # possibly also the --user and --no-index flags
>>>>>>> 776bae0d30b9fe68b2f1fa121b8faf0fa6ccdd0d
