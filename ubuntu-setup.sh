export DEBIAN_FRONTEND=noninteractive
apt-get -qq -y update
apt-get -qq -y install apt-utils curl libgl1-mesa-glx
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh --output miniconda.sh
bash miniconda.sh -b -p miniconda
export PATH=./miniconda/bin:$PATH
conda update -n base -c defaults conda
conda env create --name reborn --file environment.yml
# conda env update --name reborn --file environment.yml
source activate reborn
which pip
pip install --no-deps --editable .
