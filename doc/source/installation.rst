Installation
============

So far there is no installation scripts.  We're not at that stage yet.  For now, just add bornagain to your python path:

.. code-block:: python

	import sys
	sys.path.append("../foo/bar")


Using Anaconda and pip
----------------------

To keep the installation as painless as possible, you should probably use the `Anaconda <https://anaconda.org>`_ Python 2.7 distribution.  The dependencies below assume that `numpy` and other popular scientific modules are available.

For manipulating crystals you need to install `spglib`:

.. code-block:: bash

    pip install spglib

For running GPU simulations you need to install `pyopencl`:

.. code-block:: bash

    pip install pyopencl

For displaying data you need to install `pyqtgraph`:

.. code-block:: bash

    pip install pyqtgraph
    
The above seems to work well on Apple computers, so far.

Case studies in dependency madness
----------------------------------

Scientific Linux 6 (SL6) sucks.  Never use this Linux distribution.  It is supposed to be stable, but therefore has very old versions of many key libraries such as gcc.  To install `pyopencl` on SL6 I downloaded the pyopencl-2016.2.1 source, and then from within the directory I did the following:

.. code-block:: bash

    sudo wget -O /etc/yum.repos.d/slc6-devtoolset.repo http://linuxsoft.cern.ch/cern/devtoolset/slc6-devtoolset.repo
    sudo yum install devtoolset-2
    scl enable devtoolset-2 bash
    ./configure.py --cl-inc-dir=/usr/local/cuda/include --cl-lib-dir=/usr/local/cuda/lib64
    make install
    
Extra dependencies for building the documentation
-------------------------------------------------

This documentation is generated with `Sphinx <http://www.sphinx-doc.org>`_. If you want to build the documentation for `bornagain`, then you need to install Sphinx: 

.. code-block:: bash

    pip install sphinx
    
We also use the napoleon extension for converting Google-style docstrings:
    
.. code-block:: bash

    pip install sphinxcontrib-napoleon
    
    