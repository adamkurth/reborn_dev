#!/bin/bash

if [[ ! $(basename "$(pwd)") = 'license' ]]; then
    echo 'This script should run in the developer/license directory.'
    exit
fi

cp file_header.txt file_header_py.txt
cp file_header.txt file_header_cpp.txt
sed -i 's/#/\/\//g' file_header_cpp.txt
cp file_header.txt file_header_f90.txt
sed -i 's/#/!/g' file_header_f90.txt

for ext in py cpp f90 ; do
  header=file_header_$ext.txt
  for path in ../../reborn ../../test ../../doc; do
    for file in $( find $path -name \*$ext ); do
      if [ "$( grep 'This file is part of reborn' $file )" = "" ]; then
        echo "Prepending $header to $file"
        cat $header > tmp
        cat $file >> tmp
        mv tmp $file
      fi
    done
  done
done

rm file_header_*