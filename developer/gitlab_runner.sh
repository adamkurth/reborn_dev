#!/bin/bash

# You need to install gitlab-runner and docker for this to work.  Be sure not to use
# gitlab-runner version 11.X.Y since it has problems.  Version 13.1.1 works.

if [[ ! $(basename "$(pwd)") = 'developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit 1
fi
dock
cd ..
gitlab-runner exec docker --docker-pull-policy="if-not-present" tests
