#!/bin/bash

if [[ ! $(basename $(pwd))='developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit
fi
#
#cd ..
#pwd
#
#rm -r miniconda.sh miniconda
#
#if [[ "$OSTYPE" == "linux"* ]]; then
#  echo "Detectred Linux system"
#  condasource=https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
#elif [[ "$OSTYPE" == "darwin"* ]]; then
#  echo "Detected Mac OS system"
#  condasource=https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
#else
#  echo "Cannot recognize system"
#  exit 1
#fi
#
#[[ ! -f miniconda.sh ]] && curl $condasource --output miniconda.sh
#if [[ ! -f miniconda.sh ]]; then
#  echo "Could not dowload minicoda install script"
#  exit 1
#fi
#
#if [[ ! -d miniconda ]]; then
#  bash miniconda.sh -b -p miniconda
#  ./miniconda/bin/conda update -y -n base -c defaults conda
#fi
#
#export PATH=./miniconda/bin:$PATH
#./miniconda/bin/conda install -y pip
#yes | ./miniconda/bin/pip install numpy
#yes | ./miniconda/bin/pip install .
#cd test
#./miniconda/bin/pytest