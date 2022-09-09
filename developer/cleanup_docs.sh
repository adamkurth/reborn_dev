#!/bin/bash

if [[ ! $(basename "$(pwd)") = 'developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit
fi

docdir=$(pwd)/../doc
echo cleaning docs
rm -r $docdir/source/_static/dipole_html &> /dev/null
rm -r $docdir/build $docdir/source/api $docdir/source/auto_examples &> /dev/null
latexdir=$docdir/latex
cd $latexdir/dipole
make clean
cd $latexdir/blocks
make clean
echo done