#!/bin/bash

if [[ ! $(basename "$(pwd)") = 'developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit
fi

bash cleanup_pip.sh
bash cleanup_caches.sh
bash cleanup_compiled.sh
bash cleanup_docs.sh
