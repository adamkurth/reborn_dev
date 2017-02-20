# bornagain?

This module is all about handling diffraction under the Born approximation.  It is surely not the first module of this sort, hence the name bornagain.  It is very similar to my previous "pydiffract" module.

The aim here is to allow for flexible simulations and analysis of diffraction data.  Emphasis is placed on useability of the code for scripting tasks.  Some effort will also be dedicated to the creation of a decent diffraction viewer.  Development will be on an "as-needed" basis, but with foresight of what might come next.

# Conventions

* All units are SI, and angles are radians.
* All vectors should have a numpy shape of Nx3, in order to keep vector components close in memory.

# Installation:

I recommed using Anaconda Python 2.7, and a conda environment.  On a mac computer, I first installed Anaconda, and on that clean installation I created an environment:
``` bash
conda update conda
conda create -n bornagain
source activate bornagain
```
In order to run fast simulations on a GPU we need `pyopencl`:
```bash
conda install -c timrudge pyopencl=2014.1
```
Unfortunately, we need `spglib` for dealing with crystal symmetries (it will be removed later):
`spglib` for space group symmetry:
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
