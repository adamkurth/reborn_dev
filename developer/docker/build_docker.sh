#!/bin/bash

cp ../../environment.yml .
docker build -f Dockerfile .
rm environment.yml 
