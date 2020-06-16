#!/bin/bash

if [[ ! $(basename "$(pwd)") = 'developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit
fi

bash cleanup_caches.sh
bash cleanup_compiled.sh
bash cleanup_docs.sh
find .. -name miniconda -type d -exec rm -r {} \;