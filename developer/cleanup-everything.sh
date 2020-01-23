#!/bin/bash

if [[ ! $(basename $(pwd))='developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit
fi

echo 'Cleaning caches'
. cleanup-caches.sh

echo 'Cleaning compiled objects'
. cleanup-compiled.sh
