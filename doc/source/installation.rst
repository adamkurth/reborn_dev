Setup
=====

The following suggestions should work on Linux or MacOS.  We have not thoroughly tested bornagain on MS Windows.

In principle, setting up bornagain should be as simple as including the base directory of the bornagain git repo in
your python path.  You'll probably find that you need to install some dependencies, but all of them are known to install
easily with pip or conda (however, there may be some extra steps to enable GPU computatoins -- see the relevant section
below).  There are some pieces of Fortran code in bornagain that need to be compiled, but they
should auto-compile on first import.

While it is possible to install bornagain using the provided `setup.py` script, this is not recommended because
bornagain is under development and its API is not stable.  It is instead recommended that you keep a
clone of the bornagain git repository where you are doing your analysis or simulations so that you can reproduce your
results in the future. One way to track the *exact* version of bornagain used in your project is to add it as a
`git submodule <https://git-scm.com/book/en/v2/Git-Tools-Submodules>`_ to your project's git repository.

Dependencies
------------

As of 2020, bornagain is only tested with Python 3, because Python 2 is
`finally dead <https://www.python.org/doc/sunset-python-2/>`_ .  The `environment.yml` file lists all of the packages
that are installed upon regular testing of bornagain, which might be understood as the complete list of dependencies.
Here are the current contents of that file:

.. literalinclude:: ../../environment.yml

You can import many bornagain modules without installing *all* of these dependencies, but there is little reason to
install only a subset of them if you use a good package manager.

Complete setup from a clean Linux install
-----------------------------------------

Below are the standard steps that are tested on a clean Ubuntu Linux install every time the bornagain master branch is
updated.  The script is executed from the base directory of the git repository.

.. code-block:: bash

    apt-get -qq -y update
    apt-get -qq -y install apt-utils wget curl gfortran libgl1-mesa-glx
    curl https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh --output miniconda.sh
    bash miniconda.sh -b -p miniconda
    export PATH=./miniconda/bin:$PATH
    conda update -n base -c defaults conda
    conda env create -f bornagain-env.yml
    source activate bornagain
    python setup.py develop
    cd test
    pytest

Notes on setting up your path
-----------------------------

The main setup task is to simply ensure that Python can load the bornagain package, which means that it must be
found in a path where python searches for packages.  Here are various options that you might use to get your
path set up.  The best way is to set the appropriate environment variable so that Python looks in the right place for
bornagain.  If you are using the bash shell, you can do the following:

.. code-block:: bash

    export PYTHONPATH=$PYTHONPATH:example/path/to/bornagain/repository

If you don't want to do the above every time you open a terminal, you can add this line to your bash startup script.

As an alternative to the above, you can make a symbolic link to the bornagain package in the same directory where you
are running your script.  For example:

.. code-block:: bash

    ln -s example/path/to/bornagain/repository/bornagain path/to/where/your/script/is/located

The above method might be helpful if it is important to specify an exact copy of bornagain that must be used.

Compilation of Fortran code
---------------------------

We use the f2py program that comes with Numpy to compile Fortran code.  Although Fortran code should auto-compile on
first import, you may wish to compile manually.  This can be done using the `setup.py` script as follows:

.. code-block:: bash

    python setup.py develop

You will likely see some warnings due to Numpy or Cython -- they are probably harmless (and not our fault).

Setting up OpenCL for GPU computing
-----------------------------------

In some cases, pyopencl installs via conda without the need for any further modifications.  However, it may be necessary
to install drivers and developer toolkits for your GPU.  There are some tips in the
`pyopencl documentation <https://documen.tician.de/pyopencl/misc.html>`_ that may be helpful.  Importantly, if you have
issues you should probably start by looking in the directory `/etc/OpenCL/vendors` to see if there are any drivers
available there.  For example, you might see a driver file `pocl.icd`, which will allow you to use a CPU in absence of a
GPU, or you might find some vendor specific files such as `nvidia.icd` or `intel.icd`.  You need to make sure that these
files can be found by the pyopencl module, which probably means that you need to create a symbolic link like this:

.. code-block:: bash

    ln -s /etc/OpenCL/vendors/intel.icd ~/miniconda3/etc/OpenCL/vendors

Example setup with Miniconda
----------------------------

`Miniconda <https://conda.io/miniconda.html>`_ is a reliable and lightweight distribution of python that is known to
work well with bornagain.  The `Conda <https://conda.io/docs/>`_ package manager that comes with it makes it fast and
easy to install the dependencies of bornagain.  You might consider making a trial
`conda environment <https://conda.io/docs/user-guide/tasks/manage-environments.html>`_ to check that
everything works well, since packages like opengl, opencl, pyqt are complex and may have conflicting requirments
(however, not many problems have been noticed at least since 2019).

Assuming that you have installed conda, here's an example of how to set up a new conda environment:

.. code-block:: bash

  conda create -n bornagain -c conda-forge python=3.6 pyqt=5 scipy, h5py, future, numba, pyopencl, pocl, pytest, matplotlib, pyqtgraph, pyopengl

The only downside to the conda environment is that you need to remember to activate the environment every time you use
bornagain, like this:

.. code-block:: bash

    source activate bornagain

or like this

.. code-block:: bash

    conda activate bornagain

Note that cfelpyutils currently requires that you use pip to install.  It can be installed (after activating your
environment) as follows:

.. code-block:: bash

    pip install cfelpyutils

An even easier way to setup your environment is to use the provided environment files:

.. code-block:: bash

    conda env create -f bornagain-env.yml
    conda activate bornagain

If you don't want to use a conda environment you can just install the modules in the current environment.  For example:

.. code-block:: bash

  conda install -c conda-forge pyqt=5 scipy, h5py, future, numba, pyopencl, pocl, pytest, matplotlib, pyqtgraph, pyopengl
  pip instlall cfelpyutils

You can uninstall a conda environment as follows:

.. code-block:: bash

    conda env remove -n bornagain

Testing your setup
------------------

You can simply move into the test directory and run pytest:

.. code-block:: bash

    cd path/to/bornagain/repository
    cd test
    pytest

With some luck, you will get a nice clean output from pytest that looks like the following:

.. code-block:: bash

    ============================= test session starts ==============================
    platform darwin -- Python 3.6.7, pytest-3.9.3, py-1.7.0, pluggy-0.8.0
    rootdir: /Users/rkirian/work/projects/bornagain/test, inifile:collected 36 items

    test_analysis.py ..                                                      [  5%]
    test_clcore.py .....                                                     [ 19%]
    test_clcore_interpolations.py .                                          [ 22%]
    test_crystal.py .....                                                    [ 36%]
    test_crystfel.py .                                                       [ 38%]
    test_detector.py ....                                                    [ 50%]
    test_interpolations.py .                                                 [ 52%]
    test_minimal_dependencies.py .                                           [ 55%]
    test_numpy.py ...                                                        [ 63%]
    test_simulate_atoms.py ...                                               [ 72%]
    test_simulate_clcore.py ..                                               [ 77%]
    test_simulate_cromer_mann.py .                                           [ 80%]
    test_simulations.py .                                                    [ 83%]
    test_target_density.py ....                                              [ 94%]
    test_utils.py ..                                                         [100%]

    ========================== 36 passed in 19.55 seconds ==========================

Possible issues
---------------

**OpenCL**

If you get a runtime error involving

.. code-block:: bash

    pyopencl.cffi_cl.LogicError: clGetPlatformIDs failed:

then please read the relevant section conerning GPUs above.

**Intel GPUs**

If you have a laptop with an intel GPU you might find
`this github page <https://github.com/intel/compute-runtime/releases>`_ to be helpful.  After following the instructions
there, you should then read the relevant section conerning GPUs above.

**Scientific Linux 6**

To install `pyopencl` on SL6 I found it necessary to download the pyopencl-201X.X.X source, and then from within the
directory I did something along these lines:

.. code-block:: bash

    sudo wget -O /etc/yum.repos.d/slc6-devtoolset.repo http://linuxsoft.cern.ch/cern/devtoolset/slc6-devtoolset.repo
    sudo yum install devtoolset-2
    scl enable devtoolset-2 bash
    ./configure.py --cl-inc-dir=/usr/local/cuda/include --cl-lib-dir=/usr/local/cuda/lib64
    make install
