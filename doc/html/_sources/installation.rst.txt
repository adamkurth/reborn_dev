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

We try to make bornagain compatible with both Python 2 and 3.  For graphical interfaces, we also try to keep
compatibility with both pyqt4 and pyqt5 (which is proving to be somewhat difficult...).
If you're deciding on which version to use, here are a couple things to consider:

- At the time of this writing, the LCLS psana module requires Python 2.7.
- It doesn't appear to be easy to install both pyqt4 and pyqt5 in the same Python installation
- `Anaconda and Miniconda <https://conda.io/miniconda.html>`_ python do not support pyqt4 in Python 3.7.

It's hard to say what's best for you, but hopefully things work no matter what versions of software you use.

Example setup with Python 3
---------------------------

Supposing we want python 3 and pyqt4, we can use go with Python 3.6 installed via
`Miniconda <https://conda.io/miniconda.html>`_.

The best thing to do is probably to make a new
`conda environment <https://conda.io/docs/user-guide/tasks/manage-environments.html>`_  like this:

.. code-block:: bash

  conda create -n bornagain34 -c conda-forge python=3.6 pyqt=5 h5py numpy scipy scikit-image matplotlib ipython pytest \
  sphinx pyqtgraph pyopencl pyopengl

The only downside to the conda environment is that you need to remember to activate that environment every time you use
bornagain, like this:

.. code-block:: bash

    source activate bornagain34

If you don't want to use a conda environment you can just install the modules you need in the default environment:

.. code-block:: bash

 conda install -c conda-forge python=3.6 pyqt=5 h5py numpy scipy scikit-image matplotlib ipython pytest \
  sphinx pyqtgraph pyopencl pyopengl

In order to check if the installation worked, run the following:

.. code-block:: bash

    cd /path/to/bornagain/test
    pytest    

If you get a runtime error involving :code:`pyopencl.cffi_cl.LogicError: clGetPlatformIDs failed:` it might be necessary to manually make the path to the opencl drivers visible to pyopencl.  This is probably as simple as doing the following:

.. code-block:: bash

    cp /etc/OpenCL/vendors/nvidia.icd ~/miniconda3/etc/OpenCL/vendors

For any further issues with pyopencl, there are some helpful notes `here <https://documen.tician.de/pyopencl/misc.html>`_.

Python 2
--------

There was past success with the `Anaconda <https://anaconda.org>`_ Python 2.7 distribution.  The following usually works:

.. code-block:: bash

	conda install h5py                     # Optional for file writing
	conda install pyqt=4                   # Optional for viewing
	conda install pyqtgraph                # Optional for viewing
	conda install -c conda-forge pyopencl  # Optional for simulations
	conda install sphinx                   # Optional for building documentation

Notably, pyqtgraph seems not to work well with pyqt5, so you will need to force pyqt4 as in the above.  It is probably most reasonable to create an environment for bornagain:

.. code-block:: bash

	conda create --name bornagain
	source activate bornagain

The above will be fine if you wish to use bornagain in isolation.


Installation on Scientific Linux 6
----------------------------------

To install `pyopencl` on SL6 I found it necessary to download the pyopencl-2016.2.1 source, and then from within the directory I did something along these lines:

.. code-block:: bash

    sudo wget -O /etc/yum.repos.d/slc6-devtoolset.repo http://linuxsoft.cern.ch/cern/devtoolset/slc6-devtoolset.repo
    sudo yum install devtoolset-2
    scl enable devtoolset-2 bash
    ./configure.py --cl-inc-dir=/usr/local/cuda/include --cl-lib-dir=/usr/local/cuda/lib64
    make install

