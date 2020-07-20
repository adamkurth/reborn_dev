#!/bin/bash

# You need to install gitlab-runner and docker for this to work.  Be sure not to use
# gitlab-runner version 11.X.Y since it has problems.  Version 13.1.1 works.

if [[ ! $(basename "$(pwd)") = 'developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit 1
fi

# Notet that we use the local docker -- if the docker image changed, you'll need to push it to gitlab
# in order for it to be used.  See developer/docker/build_docker.sh .
cd ..
gitlab-runner exec docker --docker-pull-policy="if-not-present" tests

if [[ "$1" == "doc" ]]; then
  gitlab-runner exec docker --docker-pull-policy="if-not-present" doc
fi
