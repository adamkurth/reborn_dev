#!/bin/bash
if [[ ! $(basename "$(pwd)") = 'developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit
fi
[ -d logs ] || mkdir logs
PYTHONPATH=$(cd ..; pwd):$PYTHONPATH
export PYTHONPATH
cd ../test || exit
pytest "$@" | tee ../developer/logs/run_tests.log
