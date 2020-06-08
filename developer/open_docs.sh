#!/bin/bash

if [[ ! $(basename "$(pwd)") = 'developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit
fi

cd ../doc || return
./open_docs.sh
