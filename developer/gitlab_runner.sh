#!/bin/bash

# You need to install gitlab-runner and docker for this to work.  Avoid gitlab-runner version 11.X.Y since it has
# problems.  Version 13.1.1 works.

if [[ ! $(basename "$(pwd)") = 'developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit 1
fi

# Note that we use the local docker image when available.  If the local docker image is updated, you need to push it
# to gitlab in order for it to be used by the gitlab servers.  See developer/docker/build_docker.sh .
cd ..

if [[ "$1" == "test" ]]; then
	gitlab-runner exec docker --docker-pull-policy="if-not-present" tests
	exit
fi

if [[ "$1" == "doc" ]]; then
  gitlab-runner exec docker --docker-pull-policy="if-not-present" doc
  exit 
fi

gitlab-runner exec docker --docker-pull-policy="if-not-present" tests
gitlab-runner exec docker --docker-pull-policy="if-not-present" doc
