#!/bin/bash

if [[ ! $(basename "$(pwd)") = 'developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit
fi

echo cleaning docs
rm -r ../doc/build ../doc/source/api ../doc/source/auto_examples &> /dev/null
echo done