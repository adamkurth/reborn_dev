Setup
=====

The following suggestions should work on Linux or MacOS.  We have not thoroughly tested bornagain on MS Windows.

While it is possible to install bornagain using the provided setup.py script, this is not recommended because
bornagain is under development and its API is not considered to be stable.  It is instead recommended that you keep a
clone of the bornagain git
repository where you are doing your analysis or simulations so that you can reproduce your results in the future.
One way to track the *exact* version of bornagain that is
used in your analysis project is to add it as a `submodule <https://git-scm.com/book/en/v2/Git-Tools-Submodules>`_ to
your project's git repository (this can be a bit complicated, but probably worth the effort for large projects).

Quick start from a clean Linux install
--------------------------------------

Below are the standard steps that are tested on a clean Ubuntu Linux install every time the bornagain master branch is
updated.  The script should be executed from the base directory of the git repository.

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
found in a path where python searches for packages.  Here are three different options that you might use to get your
path set up:

Option (1): The best way is to set the appropriate environment variable so that Python looks in the right place for bornagain.
If you are using the bash shell, you can do the following:

.. code-block:: bash

    export PYTHONPATH=$PYTHONPATH:example/path/to/bornagain/repository

You must remember to set your path every time you use bornagain.

Option (2): You can specify the location of bornagain directly in your python scripts.  For example:

.. code-block:: python

    import sys
    sys.path.append("example/path/to/bornagain/repository")

This option is ok, but then you must remember to always run your script from the appropriate directory if the above
path is a relative path.

Option (3): You can make a symbolic link to the bornagain package in the same directory where you are running your script.  For
example:

.. code-block:: bash

    ln -s example/path/to/bornagain/repository/bornagain path/to/where/your/script/is/located

Note that "bornagain/repository/bornagain" is not a typo in the above -- in the case of a symbolic link you need to link
to the actual bornagain package, which lies *within* the git repository that is also named bornagain.  If you use this
method, you will need to run your scripts from the specific directory where the symbolic link is created.

If you do any of the above correctly, you might be able to import bornagain in the usual way:

.. code-block:: python

    import bornagain

The above might fail if you are missing dependencies, which are described below.  Another possible mode of failure is
that you have not compiled the Fortran code that bornagain uses, which is discussed in the next section.

Compilation of Fortran code
---------------------------

There are a couple of bornagain modules that use compiled Fortran code.  We use the f2py program that comes with Numpy to compile Fortran code
and create Python modules.  Although it *shouldn't* be strictly necessary to compile these to use some of the basic
bornagain classes, you should go ahead and compile them since the f2py program that is included with numpy makes this
process quite simple.  Try the following:

.. code-block:: bash

    cd developer
    ./compile-fortran.sh

You will likely see some warnings due to Numpy (these are out of our control), but so long as there are no errors you
should be all set.

Dependencies
------------

We *try* to make bornagain compatible with both Python 2 and 3.  For graphical interfaces, we also try to keep
compatibility with both pyqt4 and pyqt5.  Below are the minimal dependencies that you should need for various features.
Note that each of them have additional dependencies; you must also satisfy those dependencies.

+--------------------------------------------------------------------+-------------------------------------------------+
|The **basic classes** in bornagain require:                         |scipy, h5py, future                              |
+--------------------------------------------------------------------+-------------------------------------------------+
|Some functions will run **faster** if you install:                  |numba                                            |
+--------------------------------------------------------------------+-------------------------------------------------+
|**GPU simulations** require:                                        |pyopencl(, sometimes pocl)                       |
+--------------------------------------------------------------------+-------------------------------------------------+
|If you are **developing** bornagain you will need:                  |pytest, sphinx, sphinx_rtd_theme                 |
+--------------------------------------------------------------------+-------------------------------------------------+
|Some **visualization tools** depend on:                             |matplotlib, pyqtgraph, pyopengl                  |
+--------------------------------------------------------------------+-------------------------------------------------+
|A couple of **specialized modules** depend on:                      |psana, cfelpyutils                               |
+--------------------------------------------------------------------+-------------------------------------------------+


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

With some luck, you will get a nice clean output from pytest:

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

it might be necessary to manually make the path to the opencl drivers visible to pyopencl.  This is probably as simple
as doing the following:

.. code-block:: bash

    cp /etc/OpenCL/vendors/nvidia.icd ~/miniconda3/etc/OpenCL/vendors

If the above doesn't work, then you can try to get opencl to run on a CPU by installing the pocl package.  For issues
with pyopencl, there are some helpful notes `here <https://documen.tician.de/pyopencl/misc.html>`_.


**Scientific Linux 6**

To install `pyopencl` on SL6 I found it necessary to download the pyopencl-201X.X.X source, and then from within the
directory I did something along these lines:

.. code-block:: bash

    sudo wget -O /etc/yum.repos.d/slc6-devtoolset.repo http://linuxsoft.cern.ch/cern/devtoolset/slc6-devtoolset.repo
    sudo yum install devtoolset-2
    scl enable devtoolset-2 bash
    ./configure.py --cl-inc-dir=/usr/local/cuda/include --cl-lib-dir=/usr/local/cuda/lib64
    make install
