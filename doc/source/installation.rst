Installation
============

So far there is no installation scripts.  We're not at that stage yet.  For now, just add bornagain to your python path:

.. code-block:: python

	import sys
	sys.path.append("../foo/bar/bornagain")

Pure Anaconda Python
--------------------

To keep the installation as painless as possible, you should probably use the `Anaconda <https://anaconda.org>`_ Python 2.7 distribution.  On a clean installation the following sometimes works:

.. code-block:: bash

	conda install h5py                     # Optional for file writing
	conda install pyqt=4                   # Optional for viewing
	conda install pyqtgraph                # Optional for viewing
	conda install -c conda-forge pyopencl  # Optional for simulations
	conda install sphinx                   # Optional for building documentation

Notably, pyqtgraph seems not to work well with pyqt5, so you will need to force pyqt4 as in the above.  I've noticed a lot of dependency madness with Anaconda (cases in which installing one package breaks another) so it is probably most reasonable to create an environment for bornagain:

.. code-block:: bash

	conda create --name bornagain
	source activate bornagain

The above will be fine if you wish to use bornagain in isolation.  

Using Anaconda and pip
----------------------

Another option is to use `pip`.  For example, the following has worked in the past:

.. code-block:: bash

    pip install pyopencl
    pip install pyqtgraph

Installation on Scientific Linux 6
----------------------------------

To install `pyopencl` on SL6 I found it necessary to download the pyopencl-2016.2.1 source, and then from within the directory I did something along these lines:

.. code-block:: bash

    sudo wget -O /etc/yum.repos.d/slc6-devtoolset.repo http://linuxsoft.cern.ch/cern/devtoolset/slc6-devtoolset.repo
    sudo yum install devtoolset-2
    scl enable devtoolset-2 bash
    ./configure.py --cl-inc-dir=/usr/local/cuda/include --cl-lib-dir=/usr/local/cuda/lib64
    make install
 

System-wide installation
------------------------

Currently, there are no install scripts that will put bornagain into your python installation.  There is no setup.py file and so on.  There are two reasons for this: (1) bornagain is totally experimental still, and (2) it actually makes a lot of sense to keep a local copy where you are doing your analysis or simulations so that you can reproduce them in the future if need be.  Archiving code alongside results is good practice.

