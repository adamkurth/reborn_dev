Installation
============

The "installation" of reborn consists of including the reborn project directory in your python path, and making
sure that you have all the needed dependencies.
All of the packages that reborn depends on are known to install with conda and/or pip.
Typically, the fortran code in reborn will auto-compile the first time that you import any reborn module.
Occasionally, it will be necessary to install a fortran compiler, such as gfortran, on your system.
On rare occasions, you may need to compile the fortran code manually.
The only package that tends to be somewhat difficult is pyopencl, but you will only need this package if you plan to do
GPU computations.

Since reborn's interface is not considered "stable", it is recommended that you keep a clone of the reborn git
repository where you are doing your analysis or simulations so that you can reproduce your
results in the future.
You should consider keeping track of the exact version of reborn, down to the exact git commit, since things may change
in a month from now.
One way to track the *exact* version of reborn is to add it as a
`git submodule <https://git-scm.com/book/en/v2/Git-Tools-Submodules>`_ to your project's git repository.

Getting reborn
--------------

Assuming you have |git| on your computer, you should clone reborn on your computer:

.. code-block:: bash

    git clone https://gitlab.com/kirianlab/reborn.git

Dependencies
------------

As of 2020, reborn is only tested with Python 3, because
`Python 2 is dead <https://www.python.org/doc/sunset-python-2/>`_.
The `environment.yml` file lists all of the packages that are installed upon regular testing of reborn.
Here are the current contents of that file:

.. literalinclude:: ../../environment.yml

Some of the core modules of reborn only require |scipy| and its dependencies, but GPU simulations require |pyopencl|,
working with CrystFEL geometry files requires `cfelpyutils <https://pypi.org/project/cfelpyutils/>`__, and so on.
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
This can be done by setting the appropriate environment variable:

.. code-block:: bash

    export PYTHONPATH=$PYTHONPATH:example/path/to/reborn/repository

It might be convenient to add the above line to your `bash startup script
<https://www.gnu.org/software/bash/manual/html_node/Bash-Startup-Files.html>`_.

Compilation of Fortran code
---------------------------

Fortran code usually auto-compiles on the first import via the numpy f2py tool, but in some circumstances you may need
to compile manually.
This can be done using the ``setup.py`` script as follows:

.. code-block:: bash

    export NPY_DISTUTILS_APPEND_FLAGS=1
    export NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
    python setup.py build_ext --inplace

In old versions of |numpy|, the environment variables in the above are essential.

If you update reborn, you should re-compile the fortran code (unless you are certain that no fortran code was updated).

If the above fails, you can have a look at the ``developer/compile-fortran.sh`` script.  There are occasionally issues
caused by the mixing of dynamic libraries between different Python versions (not our fault...).

Installing reborn with pip
--------------------------

It is not recommended to install reborn with pip because you might end up with caches in places that you are unaware of,
which can cause confusion.
But if you really like the idea of installing the package, you should at least consider doing so in a way that
accommodates future changes to reborn:

.. code-block:: bash

    pip install --no-deps --editable .

You should execute the above *from the base directory* of the git repository.  The ``--editable`` flag in the above
means that you usually do not need to reinstall python code when you pull the latest updates the reborn git repository.
However, it will be necessary to reinstall if Fortran code has changed.

Setting up OpenCL for GPU computing
-----------------------------------

In some cases, |pyopencl| installs via conda without the need for additional steps.  If you have
runtime errors, check if the command ``clinfo`` indicates the presence of a GPU.
If not, then you might need to install drivers or development toolkits for your specific hardware.
If ``clinfo`` detects your GPU, then you probably just need to ensure that pyopencl can find an "Installable Client
Driver" (ICD) file.
They are often found in the directory ``/etc/OpenCL/vendors`` -- look for files with the ``.icd``
extension.
If you see the file ``pocl.icd``, then you should at least be able to use a CPU as a poor substitute for a GPU.
Ideally, you will find vendor-specific ICD files such as ``nvidia.icd`` or ``intel.icd``.
You need to make sure that these files can be found by the |pyopencl| module, which *probably* means that you need to
create a symbolic link like this:

.. code-block:: bash

    ln -s /etc/OpenCL/vendors/intel.icd ~/miniconda3/etc/OpenCL/vendors

If the above fails, read through the tips in the
`pyopencl documentation <https://documen.tician.de/pyopencl/misc.html>`_.

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

For pyopencl, it might be helpful to install pocl:

.. code-block:: bash

    apt-get install pocl-opencl-icd

For pyopengl, the following might help:

.. code-block:: bash

    apt-get install libgl1-mesa-glx


Mac OS notes
------------

It might help to install xcode, Homebrew, or similar, to get gfortran, for example.

Windows 10 Notes
----------------

The best option on Windows is probably to use a virtual machine such as VirtualBox to get a proper Linux environment.

Another option for Windows 10 is to install Microsoft's Ubuntu subsystem.
There are issues with displaying windows when using the Ubuntu subsystem, and one work-around is
to install VcXsrv.

1) Download the Ubuntu app from the Windows app store
2) Open the Windows Powershell, run as administrator
3) Run this line and restart your computer

.. code-block:: bash

    Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux

Note that you will likely need to ``cd /mnt/c/../..`` to change to the c-drive (or whichever drive you wish).

From here, you need to install VcXsrv Windows X Server. Here is a link to the 2020-01-12 version:
https://sourceforge.net/projects/vcxsrv/
It's a little annoying, but you'll have to manually open the software and go through all the default options
every time you reboot your computer.

4) Download and run VcXsrv and run the installer with all the default settings. Make sure to choose the 'multiple
   windows' options.

5) In the Ubuntu app, install imagemagick

.. code-block:: bash

    sudo apt install imagemagick

6) In the Ubuntu terminal, run this line

.. code-block:: bash

    echo "export DISPLAY=localhost:0.0" >> ~/.bashrc && source ~/.bashrc

To install Python and all the important stuff, go to Anaconda.com and download the LINUX version of the software suite.
Then in your Ubuntu terminal, navigate to the download and install it using this command

.. code-block:: bash

    bash /your/file/path/Anaconda2-2019.10-Linux-x86_64.sh

Make you sure you change your file path and double check that the download file is the most up to date Linux
installation file. Follow through with all the default installation settings and restart your terminal once the download
is complete.  After all of that is complete, you should have the most up-to-date python and ipython versions. You can
download all the packages you need by running conda install [package].

To get submodules to work for Windows, follow this guide:

1) In your ~/.ssh/ folder, add a new text file and name it 'config'.

.. code-block:: bash

    sudo nano config

2)  In that file, add the following text:

.. code-block:: bash

    AddressFamily inet

3)  In your repository, do the following. Note: 'B' in the commit message should be changed to the repo you're adding
bornagain to.

.. code-block:: bash

    git submodule add git@gitlab.com:rkirian/bornagain.git
    git submodule update --remote

This should work fine from here, but you may need to add a symbolic link from the location of your script to the bornagain/reborn folder in order to get things working.

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
