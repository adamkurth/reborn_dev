#!/bin/bash
#[ -d logs ] || mkdir logs
if [[ ! $(basename "$(pwd)") = 'developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit
fi
#exec > >(tee -ia logs/rebuild_docs.log)
#exec 2> >(tee -ia logs/rebuild_docs.error.log)
bash cleanup_everything.sh
bash update_docs.sh
