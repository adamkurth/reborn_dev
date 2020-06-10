#!/bin/bash

if [[ ! $(basename "$(pwd)") = 'developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit
fi

bash cleanup-pip.sh
bash cleanup-caches.sh
bash cleanup-compiled.sh
bash cleanup-docs.sh
