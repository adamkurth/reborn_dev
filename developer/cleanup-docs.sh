#!/bin/bash

if [[ ! $(basename "$(pwd)") = 'developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit
fi

echo 'Cleaning sphinx output'
rm -r ../doc/html ../doc/build ../doc/source/api
