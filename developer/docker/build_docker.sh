#!/bin/bash

cp ../../environment.yml .
docker build -f Dockerfile .
rm environment.yml
# Don't forget to docker tag and docker push!