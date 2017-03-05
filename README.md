# bornagain?

This module is all about handling diffraction under the Born approximation.  It is surely not the first module of this sort, hence the name bornagain.  It is very similar to my previous "pydiffract" module.

The aim here is to allow for flexible simulations and analysis of diffraction data.  Emphasis is placed on useability of the code for scripting tasks.  Some effort will also be dedicated to the creation of a decent diffraction viewer.  Development will be on an "as-needed" basis, but with foresight of what might come next.

# Conventions

* All units are SI, and angles are radians.
* All vectors should have a numpy shape of Nx3, in order to keep vector components close in memory.

# Installation:

I recommend using Anaconda Python 2.7, and a conda environment.  On a Mac or Linux computer, I first installed Anaconda by doing something like this:
``` bash
wget https://repo.continuum.io/archive/Anaconda2-4.3.0-Linux-x86_64.sh
bash Anaconda2-4.3.0-Linux-x86_64.sh
conda update conda
```
On that clean installation I created a conda environment:
``` bash
conda create -n bornagain
source activate bornagain
```
In order to run fast simulations on a GPU we need `pyopencl`:
```bash
pip install pyopencl
```
Unfortunately, we need `spglib` for dealing with crystal symmetries (it will be removed later):
```bash
pip install spglib
```
So far these are not strictly needed, but will probably become dependencies:
`pyqtgraph` for viewing:
```bash
conda install -c ufechner pyqtgraph=0.9.10
```
`pyopengl` for plotting 3d graphics:
```bash
conda install pyopengl
```

# Notes on installation on Scientific Linux 6
This is very incomplete... but in part the following was needed to deal with the very old version of gcc on SL6 (don't use SL6 ever again...)
```bash
pip install mako
[cd pyopencl-2016.2.1; downloaded source...]
sudo wget -O /etc/yum.repos.d/slc6-devtoolset.repo http://linuxsoft.cern.ch/cern/devtoolset/slc6-devtoolset.repo
sudo yum install devtoolset-2
scl enable devtoolset-2 bash
rm ./siteconf.py ; ./configure.py --cl-inc-dir=/usr/local/cuda/include --cl-lib-dir=/usr/local/cuda/lib64; make install
```

# Notes for developers
* This project follows the [PEP 8](https://www.python.org/dev/peps/pep-0008) style guide for python code.



