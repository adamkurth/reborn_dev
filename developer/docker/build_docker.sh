#!/bin/bash

if [[ ! $(basename "$(pwd)") = 'docker' ]]; then
    echo 'This script should run in the developer/docker directory.'
    exit 1
fi

cp ../../environment.yml .
docker -v build -f Dockerfile -t registry.gitlab.com/kirianlab/reborn/ubuntu-20.04:latest .
rm environment.yml

# This makes the image available to gitlab runners
if [[ "$1" == "push" ]]; then
    docker login registry.gitlab.com
    docker push registry.gitlab.com/kirianlab/reborn/ubuntu-20.04:latest
fi