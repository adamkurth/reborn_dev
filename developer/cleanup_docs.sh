#!/bin/bash

if [[ ! $(basename "$(pwd)") = 'developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit
fi


echo cleaning docs
rm -r ../doc/build ../doc/source/api ../doc/source/auto_examples &> /dev/null
latexdir=$(pwd)/../doc/latex
cd $latexdir/dipole
make clean
cd $latexdir/blocks
make clean
echo done