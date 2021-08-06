#!/bin/bash

if [[ ! $(basename "$(pwd)") = 'license' ]]; then
    echo 'This script should run in the developer/license directory.'
    exit
fi

for file in $(find ../../reborn -name '*.py'); do
  cat file_header.txt > tmp
  cat $file >> tmp
  mv tmp $file
done