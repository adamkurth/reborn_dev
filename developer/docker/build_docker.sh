#!/bin/bash

cp ../../environment.yml .
docker build -f Dockerfile .
rm environment.yml
# docker login registry.gitlab.com
# docker tag <hash>
# docker push
