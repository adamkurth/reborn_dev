#!/bin/bash
cd ../doc/examples || exit
export PYTHONPATH=$PYTHONPATH:../..
for file in *.py; do
  clear
  echo "$file"
  python "$file"
  sleep 2
done