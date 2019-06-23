Setting up bornagain
====================

"Installation" of bornagain
---------------------------

There are no install scripts (e.g. setup.py) that will automatically add bornagain to your python installation.
Since bornagain is changing frequently at this time, it is recommended that you keep a local copy of bornagain
where you are doing your analysis or simulations so that you can reproduce them in the future if need be.  Archiving
code alongside results is good practice anyways.

The main setup task is to simply ensure that Python can load the bornagain package.  Here are three different
suggestions for how to get your path set up:

1) Within your shell, you can set an environment variable so that Python looks in the right place for bornagain.
If you are using the bash shell, you can do the following:

.. code-block:: bash

    export PYTHONPATH=example/path/to/bornagain

2) You can specify the location of bornagain directly in your python path.  For example:

.. code-block:: python

    import sys
    sys.path.append("example/path/to/bornagain")

3) You can make a symbolic link to the bornagain package in the same directory where you are running your script.  For
example:

.. code-block:: bash

    ln -s example/path/to/bornagain/bornagain path/to/where/your/script/is/located

Note that "bornagain/bornagain" is not a typo in the above -- in the case of a symbolic link you need to link to the
actual bornagain package, which is not the same as the git repository bornagain that contains the bornagain package.

If you do any of the above correctly, you should then be able to import bornagain in the usual way:

.. code-block:: python

    import bornagain


Compilation of Fortran code
---------------------------

There are a couple of bornagain modules that rely on Fortran.  In principle, they are not necessary, but they will speed
up some routines.  We use the f2py program to compile Fortran code and create Python modules.  Most likely, you can
simply do the following:

.. code-block:: bash

    cd developer
    ./compile-fortran.sh

You will see some warnings (due to Numpy -- this is out of our control).  So long as there are no errors you should be
all set.

Dependencies
------------

The most basic classes in bornagain should work so long as you have the following dependencies installed:

* h5py
* numpy
* scipy
* future

Some functions will run faster if you install:

* numba

If you want to run simulations, you'll need

* pyopencl

If you are developing bornagain you will need

* pytest
* sphinx

There are some visualization tools that depend on:

* matplotlib
* pyqtgraph
* pyopengl

A couple of specialized packages are used for dealing with LCLS XTC data and CrystFEL geometry files:

* psana
* cfelpyutils

We try to make bornagain compatible with both Python 2 and 3.  For graphical interfaces, we also try to keep
compatibility with both pyqt4 and pyqt5.

Example setup
-------------

`Miniconda <https://conda.io/miniconda.html>`_ is a reliable and lightweight distribution of python that is known to
work well with bornagain.  The `Conda <https://conda.io/docs/>`_ package manager that comes with it makes it fast and
easy to install the dependencies of bornagain.  You might
consider making a `conda environment <https://conda.io/docs/user-guide/tasks/manage-environments.html>`_ to check that
everything works well, since packages like opengl, opencl, pyqt have had conflicting requirments in the past (however,
not many problems have been noticed since 2019).

Assuming that you've installed conda, here's an example of how to set up a new conda environment:

.. code-block:: bash

  conda create -n bornagain -c conda-forge python=3.6 pyqt=5 h5py numpy scipy scikit-image \
  matplotlib ipython pytest sphinx pyqtgraph pyopencl pyopengl future

The only downside to the conda environment is that you need to remember to activate the environment every time you use
bornagain, like this:

.. code-block:: bash

    source activate bornagain

Note that cfelpyutils currently requires that you use pip to install.  It can be installed (after activating your
environment) as follows:

.. code-block:: bash

    pip install cfelpyutils

An even easier way to setup your environment is to use the provided environmen files:

.. code-block:: bash

    conda env create -f bornagain3-env.yml
    conda activate bornagain3

If you don't want to use a conda environment you can just install the modules in the current environment.  For example:

.. code-block:: bash

  conda install -c conda-forge python=3.6 pyqt=5 h5py numpy scipy scikit-image matplotlib ipython pytest \
  sphinx pyqtgraph pyopencl pyopengl future
  pip instlall cfelpyutils

You can check if you've got all the dependencies sorted out by running the following:

.. code-block:: bash

    cd bornagain/test
    pytest

You can uninstall a conda environment as follows:

.. code-block:: bash

    conda env remove -n bornagain3


Possible issues
---------------

**OpenCL**

If you get a runtime error involving

.. code-block:: bash

    pyopencl.cffi_cl.LogicError: clGetPlatformIDs failed:

it might be necessary to manually make the path to the opencl drivers visible to pyopencl.  This is probably as simple as doing the following:

.. code-block:: bash

    cp /etc/OpenCL/vendors/nvidia.icd ~/miniconda3/etc/OpenCL/vendors

For any further issues with pyopencl, there are some helpful notes `here <https://documen.tician.de/pyopencl/misc.html>`_.

If you get a runtime error like this

.. code-block:: bash

    pyopencl._cl.LogicError: clGetPlatformIDs failed: PLATFORM_NOT_FOUND_KHR

you should try installing the package pocl.  I don't know why this fixes the problem but it has worked on a couple
of Linux systems thus far.


**Scientific Linux 6**

To install `pyopencl` on SL6 I found it necessary to download the pyopencl-201X.X.X source, and then from within the
directory I did something along these lines:

.. code-block:: bash

    sudo wget -O /etc/yum.repos.d/slc6-devtoolset.repo http://linuxsoft.cern.ch/cern/devtoolset/slc6-devtoolset.repo
    sudo yum install devtoolset-2
    scl enable devtoolset-2 bash
    ./configure.py --cl-inc-dir=/usr/local/cuda/include --cl-lib-dir=/usr/local/cuda/lib64
    make install