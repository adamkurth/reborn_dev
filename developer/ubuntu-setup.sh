#!/bin/bash

apt-get -qq -y update
apt-get -qq -y install apt-utils curl libgl1-mesa-glx
echo getting miniconda
curl https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -vvv --output miniconda.sh
echo try miniconda.sh
bash miniconda.sh -b -p miniconda
bash miniconda.sh --help
bash miniconda.sh -h
export PATH=./miniconda/bin:$PATH
pwd
ls
ls ./miniconda
ls ./miniconda/bin
echo where is conda?
conda update -n base -c defaults conda
conda env create -f environment.yml
source activate reborn
python setup.py develop
