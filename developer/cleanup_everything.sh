#!/bin/bash

if [[ ! $(basename "$(pwd)") = 'developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit
fi

echo cleaning everything
bash cleanup_caches.sh
bash cleanup_compiled.sh
bash cleanup_docs.sh
echo cleaned everything