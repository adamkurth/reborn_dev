Installation
============

System-wide installation
------------------------

Currently, there are no install that will automatically add bornagain to your python installation.  It makes a lot of sense to simply keep a local copy of bornagain where you are doing your analysis or simulations so that you can reproduce them in the future if need be, especially since bornagain continues to change often.  Archiving code alongside results is good practice anyways.  

If the bornagain directory is not in your working directory, you might need to add bornagain to your python path by doing something like this:

.. code-block:: python

	import sys
	sys.path.append("../example/path/to/bornagain")


Python 3
--------

There has been recent success with Python 3.6.4, installed via `Miniconda <https://conda.io/miniconda.html>`_.  The following has worked on both Mac and Linux computers:

.. code-block:: bash

 conda install ipython
 conda install pytest
 conda install sphinx
 conda install scipy
 conda install scikit-image
 conda install matplotlib
 conda install h5py
 conda install pyqtgraph
 conda install pyopengl
 conda install -c conda-forge pyopencl
 conda install -c conda-forge pocl
    
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

