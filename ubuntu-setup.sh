#!/bin/bash

export DEBIAN_FRONTEND=noninteractive  # Avoid interactions on gitlab runners
apt-get -qq -y update
apt-get -qq -y install apt-utils curl libgl1-mesa-glx
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh --output miniconda.sh
bash miniconda.sh -b -p miniconda
# export PATH=./miniconda/bin:$PATH
miniconda/bin/conda init bash
bash ~/.bashrc
conda update -n base -c defaults conda
conda env create --name reborn --file environment.yml
conda activate reborn
pip install --no-deps --editable .
#python setup.py develop
echo "$PATH"
command -v pytest