#!/bin/bash

cd ../doc/examples
export PYTHONPATH=$PYTHONPATH:../..
for file in *.py; do
  clear
  echo $file
  python "$file"
  sleep 2
done