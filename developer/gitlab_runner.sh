#!/bin/bash

# You need to install gitlab-runner and docker for this to work.  Be sure not to use
# gitlab-runner version 11.X.Y since it has problems... I use version 13.1.1

if [[ ! $(basename "$(pwd)") = 'developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit 1
fi

cd ..
gitlab-runner exec docker tests
