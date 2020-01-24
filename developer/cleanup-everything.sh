#!/bin/bash

if [[ ! $(basename $(pwd))='developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit
fi

. cleanup-caches.sh
. cleanup-compiled.sh
. cleanup-docs.sh
