Installation
============

System-wide installation
------------------------

There are no install scripts (e.g. setup.py) that will automatically add bornagain to your python installation.
Since bornagain is changing frequently at this time, it is recommended that you keep a local copy of bornagain
where you are doing your analysis or simulations so that you can reproduce them in the future if need be.  Archiving
code alongside results is good practice anyways.

If the bornagain package directory is not in your working directory, you might need to add bornagain to your python path
by doing something like this:

.. code-block:: python

	import sys
	sys.path.append("../example/path/to/bornagain")

Another option is to make a symbolic link to the bornagain package, like this:

.. code-block:: bash

    ln -s path/to/bornagain/bornagain path/where/your/script/is

If you use a symbolic link as above, you can simply add an import statement in your script

.. code-block:: python

    import bornagain


Dependencies
------------

The current dependencies of bornagain are:

* h5py
* numpy
* scipy
* matplotlib
* ipython
* pytest
* sphinx
* pyqtgraph
* pyopencl
* pyopengl
* future
* cfelpyutils

You can use bornagain with only a subset of of the above dependencies (e.g. skip pyopencl if you don't want to simulate,
and skip sphinx if you won't be modifying the documentation, skip pyqtgraph if you won't use those GUIs, etc.),
but it's usually not difficult to install all of them for completeness.

We try to make bornagain compatible with both Python 2 and 3.  For graphical interfaces, we also try to keep
compatibility with both pyqt4 and pyqt5 (which is proving to be somewhat difficult...). If you're deciding on which
version to use, here are a couple things to consider:

- At the time of this writing, the LCLS psana module requires Python 2.7.
- It doesn't appear to be easy to install both pyqt4 and pyqt5 in the same Python installation
- `Anaconda and Miniconda <https://conda.io/miniconda.html>`_ python do not support pyqt4 in Python 3.7.

It's hard to say what's best for you, but hopefully things work no matter what versions of software you use.

Example setup
-------------

`Miniconda <https://conda.io/miniconda.html>`_ is a reliable and lightweight distribution of python.  The
`Conda <https://conda.io/docs/>`_ package manager that comes with it makes it fast and easy to set up Python.  You might
consider making a `conda environment <https://conda.io/docs/user-guide/tasks/manage-environments.html>`_ to check that
everything works well, since packages like opengl, opencl, pyqt seem to have complicated, and occasionally conflicting
requirments.

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


Possible installation issues
----------------------------

**OpenCL**

If you get a runtime error involving :code:`pyopencl.cffi_cl.LogicError: clGetPlatformIDs failed:` it might be necessary to manually make the path to the opencl drivers visible to pyopencl.  This is probably as simple as doing the following:

.. code-block:: bash

    cp /etc/OpenCL/vendors/nvidia.icd ~/miniconda3/etc/OpenCL/vendors

For any further issues with pyopencl, there are some helpful notes `here <https://documen.tician.de/pyopencl/misc.html>`_.


**Scientific Linux 6**

To install `pyopencl` on SL6 I found it necessary to download the pyopencl-201X.X.X source, and then from within the
directory I did something along these lines:

.. code-block:: bash

    sudo wget -O /etc/yum.repos.d/slc6-devtoolset.repo http://linuxsoft.cern.ch/cern/devtoolset/slc6-devtoolset.repo
    sudo yum install devtoolset-2
    scl enable devtoolset-2 bash
    ./configure.py --cl-inc-dir=/usr/local/cuda/include --cl-lib-dir=/usr/local/cuda/lib64
    make install