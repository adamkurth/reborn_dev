#!/bin/bash
pwd
if [[ ! $(basename "$(pwd)") = 'developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit
fi

cd ../doc || return
./update_docs.sh "$@"
