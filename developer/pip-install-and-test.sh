#!/bin/bash

if [[ ! $(basename $(pwd))='developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit
fi

cd ..
pwd

if [[ "$OSTYPE" == "linux"* ]]; then
  echo "Detectred Linux system"
  condasource=https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
elif [[ "$OSTYPE" == "darwin"* ]]; then
  echo "Detected Mac OS system"
  condasource=https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
else
  echo "Cannot recognize system"
  exit 1
fi


[[ ! -f miniconda.sh ]] && curl $condasource --output miniconda.sh
if [[ ! -f miniconda.sh ]]; then
  echo "Could not dowload minicoda install script"
  exit 1
fi
[[ ! -d miniconda ]] && source miniconda.sh -b -p miniconda
ls miniconda
export PATH=./miniconda/bin:$PATH
./miniconda/bin/conda install pip
yes | ./miniconda/bin/pip install .
cd test
./miniconda/bin/pytest