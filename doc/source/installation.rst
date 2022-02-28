Installation
============

The "installation" of reborn consists of including the reborn project directory in your python path, and making
sure that you have all the needed dependencies.
All of the packages that reborn depends on are known to install with conda and/or pip.
Fortran code included in reborn should auto-compile upon first import.
The only package that tends to be somewhat difficult is pyopencl, which is only needed if you plan to do GPU
computations.

Although we try to maintain backward compatibility as we develop reborn, it is under active development and the API is
subject to change.  If reborn is used to  produce important results that you need to replicate in the future, you should
consider keeping track of the version of reborn, down to the exact git commit.  One way to do this is to add reborn as a
`git submodule <https://git-scm.com/book/en/v2/Git-Tools-Submodules>`_ to your project's git repository.

Getting reborn
--------------

Assuming you have |git| on your computer, you should clone reborn on your computer:

.. code-block:: bash

    git clone https://gitlab.com/kirianlab/reborn.git

Dependencies
------------

As of 2020, reborn is only tested with Python 3.
`Never use Python 2 <https://www.python.org/doc/sunset-python-2/>`_.
The `environment.yml` file lists all of the packages that are installed upon regular testing of reborn.
Here are the current contents of that file:

.. literalinclude:: ../../environment.yml

Some of the core modules of reborn only require |scipy| and its dependencies, but GPU simulations require |pyopencl|,
viewers require |pyqt5|, and so on.
If you use a good package manager you might as well install all of the above dependencies.

Setting up Python with Miniconda
--------------------------------

|Miniconda| is a reliable and lightweight distribution of python that is known to work well with reborn.
The `Conda <https://conda.io/docs/>`_ package manager that comes with it makes it fast and easy to install and maintain
the dependencies of reborn.
It is recommended that you first make a trial
`conda environment <https://conda.io/docs/user-guide/tasks/manage-environments.html>`_ to check that there are no
conflicts between dependencies.
The simplest way to setup a conda environment with all the needed reborn dependencies is to execute the following
command in the base directory of the reborn git repository:

.. code-block:: bash

    conda env create --name reborn --file environment.yml

The above line will create the conda environment named ``reborn`` (you may choose a different name if you wish).  You
will need to activate that environment whenever you want to use it:

.. code-block:: bash

    source activate reborn

If you wish, you can also install the dependencies into the default ``base`` conda environment (or another environment
that already exists):

.. code-block:: bash

    conda env update --name base --file environment.yml

Including reborn in your python path
------------------------------------

You do not need to "install" reborn; just add the reborn repository to the python search path.
This can be done by setting the appropriate environment variable.  For example, in the |bash| shell:

.. code-block:: bash

    export PYTHONPATH=$PYTHONPATH:example/path/to/reborn/repository

It might be convenient to add the above line to your `bash startup script
<https://www.gnu.org/software/bash/manual/html_node/Bash-Startup-Files.html>`_.

Compilation of Fortran code
---------------------------

Fortran code usually auto-compiles on the first import via the numpy |f2py| tool, but in some circumstances you may need
to compile manually.  If reborn fails to import the fortran module, you can have a look at the
`compile-fortran.sh <https://gitlab.com/kirianlab/reborn/-/blob/master/developer/compile_fortran.sh>`_ script.

Installing reborn with pip
--------------------------

It is not recommended to install reborn with pip because you might end up with caches in places that you are unaware of.
But if you really like the idea of installing reborn, you should at least consider doing so in a way that
accommodates future changes:

.. code-block:: bash

    pip install --no-deps --editable .

Setting up OpenCL for GPU computing
-----------------------------------

In some cases, |pyopencl| installs via conda without the need for additional steps.  If you have runtime errors, check
if the command ``clinfo`` indicates the presence of a GPU.  If not, then you might need to install drivers or
development toolkits for your specific hardware.  If ``clinfo`` detects your GPU, then you probably just need to ensure
that the pyopencl package can find an "Installable Client Driver" (ICD) file.  These files are often found in the
directory ``/etc/OpenCL/vendors`` -- they should have a ``.icd`` extension.  If you see the file ``pocl.icd``, then you
should at least be able to use a CPU as a poor substitute for a GPU.  Ideally, you will find vendor-specific ICD files
such as ``nvidia.icd`` or ``intel.icd``.  Often times you can help |pyopencl| find the ICD files by creating a symbolic
link, for example like this:

.. code-block:: bash

    ln -s /etc/OpenCL/vendors/intel.icd ~/miniconda3/etc/OpenCL/vendors

If the above fails, read through the tips in the
`pyopencl documentation <https://documen.tician.de/pyopencl/misc.html>`_.

Installing pyvkfft for performing FFTs on GPUs
----------------------------------------------

`vkfft <https://github.com/DTolm/VkFFT/>`_ is a GPU-accelerated multi-dimensional Fast Fourier Transform library
supporting many backends (Vulkan, CUDA, HIP and OpenCL). The reborn package does not depend on vkfft, but it may be
helpful for your application.  To get it to install (as of July 2021) the instructions below should be of help.

.. code-block:: bash

    conda env create --name name_of_your_environment --file environment.yml
    conda install cython
    conda install -c conda-forge pycuda
    conda install -c conda-forge ocl-icd-system
    pip install pyvkfft

On ASU's Agave cluster, to request an interactive node with GPUs you can do something like

.. code-block:: bash

    interactive -p gpu -q wildfire -t 60 --gres=gpu:1 

Testing your setup
------------------

You can simply move into the test directory and run |pytest|:

.. code-block:: bash

    cd path/to/reborn/repository
    cd test
    pytest

Linux notes
-----------

If you need a fortran compiler:

.. code-block:: bash

    apt-get install gfortran

For pyopengl, the following might help:

.. code-block:: bash

    apt-get install libgl1-mesa-glx


Mac OS notes
------------

The Linux notes mostly apply to Mac OS also.  Presumably you will need to install xcode and use homebrew, conda,
or similar to get the gfortran compiler.

Windows 10 Notes
----------------

The best option on Windows is probably to use a virtual machine such as VirtualBox to get a proper Linux environment.
Another option is to use the Linux subsystem on Windows 10, as discussed :ref:`here <windows_anchor>` .

Possible issues
---------------

**OpenCL**

If you get a runtime error involving

.. code-block:: bash

    pyopencl.cffi_cl.LogicError: clGetPlatformIDs failed:

then please read the relevant section concerning "Installable Client Drivers" (ICDs) above.

If you get a segmentation fault immediately when you try to use pyopencl, you might need to try a different ICD.  For
example, if the ICD set up by conda fails, try installing one using apt.

**Intel GPUs on Ubuntu**

If you have an Ubuntu-like OS and a laptop with an intel GPU you might find
`this github page <https://github.com/intel/compute-runtime/releases>`_ helpful.  After following the instructions
there, you should then read the relevant section concerning GPUs above.

**Scientific Linux 6**

To install `pyopencl` on SL6 I found it necessary to download the pyopencl-201X.X.X source, and then from within the
directory I did something along these lines:

.. code-block:: bash

    sudo wget -O /etc/yum.repos.d/slc6-devtoolset.repo http://linuxsoft.cern.ch/cern/devtoolset/slc6-devtoolset.repo
    sudo yum install devtoolset-2
    scl enable devtoolset-2 bash
    ./configure.py --cl-inc-dir=/usr/local/cuda/include --cl-lib-dir=/usr/local/cuda/lib64
    make install
