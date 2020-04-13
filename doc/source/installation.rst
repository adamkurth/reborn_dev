Getting Started
===============

In principle, setting up reborn should be as simple as including the base directory of the reborn git repository in
your python path.  You'll probably find that you need to install some dependencies, but all of them are known to install
easily with pip or conda (however, there may be some extra steps to enable GPU computations -- see the relevant section
below).  There are some pieces of Fortran code in reborn that need to be compiled, but they should auto-compile on first
import.

While it is possible to install reborn using the provided `setup.py` script, this is not recommended because
reborn is under development and its API is not stable.  It is instead recommended that you keep a
clone of the reborn git repository where you are doing your analysis or simulations so that you can reproduce your
results in the future. One way to track the *exact* version of reborn used in your project is to add it as a
`git submodule <https://git-scm.com/book/en/v2/Git-Tools-Submodules>`_ to your project's git repository.

The first thing you will need to do is use git to clone reborn on your computer:

.. code-block:: bash

    git clone git@gitlab.com:rkirian/bornagain.git reborn

Note that reborn was formerly known as bornagain, which is why the above command renames the repository.  Eventually,
the link will be fixed so that this is not necessary.

Dependencies
------------

As of 2020, reborn is only tested with Python 3, because Python 2 is
`finally dead <https://www.python.org/doc/sunset-python-2/>`_ .  The `environment.yml` file lists all of the packages
that are installed upon regular testing of reborn, which might be understood as the complete list of dependencies.
Here are the current contents of that file:

.. literalinclude:: ../../environment.yml

You can import many reborn modules without installing *all* of these dependencies, but there is little reason to
install only a subset of them if you use a good package manager.

Miniconda
---------

`Miniconda <https://conda.io/miniconda.html>`_ is a reliable and lightweight distribution of python that is known to
work well with reborn.  The `Conda <https://conda.io/docs/>`_ package manager that comes with it makes it fast and
easy to install the dependencies of reborn.  You might consider making a trial
`conda environment <https://conda.io/docs/user-guide/tasks/manage-environments.html>`_ to check that
everything works well, since packages like opengl, opencl, pyqt are complex and may have conflicting requirments
(however, not many problems have been noticed at least since 2019).  The simplest way to setup a proper conda
environment is to execute the following (in the base directory of the reborn respository):

.. code-block:: bash

    conda env create -f environment.yml

The above line will create a conda environment named `reborn`, and you will need to activate that environment in the
following way:

.. code-block:: bash

    conda activate reborn

The `conda activate` often does not work, in which case you might try the following instead:

.. code-block:: bash

    source activate reborn

If you don't want to create a new environment named `reborn`, you can instead add all of the packages in the
`environment.yml` file to the default `base` environment (or to any other environment you specify with the `name`
flag):

.. code-block:: bash

    conda env update --name base --file environment.yml

You can of course install all of the necessary packages manually by other means.

Linux notes
-----------

Below are the standard steps that are tested on a clean Ubuntu Linux install every time the reborn master branch is
updated.  The script `developer/ubuntu-install.sh` is executed from the base directory of the git repository.  Here
are the current contents of that script:

.. literalinclude:: ../../developer/ubuntu-setup.sh

Mac OS notes
------------

Mostly the same procedure as Linux, except that you should install Miniconda according to the instructions for 
Mac OS.


Windows 10 Notes
----------------

One method that is known to work for Windows 10 is to install Microsoft's Ubuntu subsystem.
There are issues with displaying windows when using the Ubuntu subsystem, and one work-around is
to install VcXsrv.

1) Download the Ubuntu app from the Windows app store
2) Open the Windows Powershell, run as administrator
3) Run this line and restart your computer

.. code-block:: bash

    Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux

Note that you will likely need to cd /mnt/c/../.. to change to the c-drive (or whichever drive you wish)

From here, you need to install VcXsrv Windows X Server. Here is a link to the 2020-01-12 version: https://sourceforge.net/projects/vcxsrv/
It's a little annoying, but you'll have to manually open the software and go through all the default options 
everytime you reboot your computer. 

4) Download and run VcXsrv and run the installer with all the default settings. Make sure to choose the 'multiple windows' options.
5) In the Ubuntu app, install imagemagick

.. code-block:: bash

    sudo apt install imagemagick

6) In the Ubuntu terminal, run this line

.. code-block:: bash

    echo "export DISPLAY=localhost:0.0" >> ~/.bashrc && source ~/.bashrc



To install Python and all the important stuff, go to Anaconda.com and download the LINUX version of the software suite. Then in your Ubuntu terminal,
navigate to the download and install it using this command

.. code-block:: bash

    bash /your/file/path/Anaconda2-2019.10-Linux-x86_64.sh

Make you sure you change your file path and double check that the download file is the most up to date Linux installation file. Follow through with
all the default installation settings and restart your terminal once the download is complete.  After all of that is complete, you should have the 
most up-to-date python and ipython versions. You can download all the packages you need by running conda install [package]. 



To get submodules to work for Windows, follow this guide:

1) In your ~/.ssh/ folder, add a new text file and name it 'config'.

.. code-block:: bash

    sudo nano config

2)  In that file, add the follwing text: 

.. code-block:: bash

    AddressFamily inet

3)  In your repository, do the following. Note: 'B' in the commit messsage should be changed to the repo you're adding bornagain to. 

.. code-block:: bash

    git submodule add git@gitlab.com:rkirian/bornagain.git
    git submodule update --remote

This should work fine from here, but you may need to add a symbolic link from the location of your script to the bornagain/reborn folder in order to get things working. 







Notes on setting up your path
-----------------------------

The main setup task is to simply ensure that Python can import the reborn package, which means that it must be
found in a path where python searches for packages.  If you don't already know how to do this, here are some options
that you might use to get your path set up.  The best way is to set the appropriate environment variable so that Python
looks in the right place for reborn.  If you are using the bash shell, you can do the following:

.. code-block:: bash

    export PYTHONPATH=$PYTHONPATH:example/path/to/reborn/repository

If you don't want to do the above every time you open a terminal, you can add this line to your `bash startup script
<https://www.gnu.org/software/bash/manual/html_node/Bash-Startup-Files.html>`_.

As an alternative to the above, you can make a symbolic link to the reborn package in the same directory where you
are running your script.  For example:

.. code-block:: bash

    ln -s example/path/to/reborn/repository/reborn path/to/where/your/script/is/located

If you wish, you can also install reborn:

.. code-block:: bash

    python setup.py install

but this is not tested and is not advised as noted above.

Compilation of Fortran code
---------------------------

We use the f2py program that comes with Numpy to compile Fortran code.  Although Fortran code should auto-compile on
first import, you may wish to compile manually.  This can be done using the `setup.py` script as follows:

.. code-block:: bash

    python setup.py develop

You will likely see some warnings due to Numpy or Cython -- they are probably harmless (and not our fault).

Setting up OpenCL for GPU computing
-----------------------------------

In some cases, pyopencl installs via conda without the need for any further modifications.  However, if you have runtime
errors, it may be necessary to install drivers and/or developer toolkits for your GPU.  You should read through the tips
in the `pyopencl documentation <https://documen.tician.de/pyopencl/misc.html>`_.  As discussed there, you should
probably start by looking in the directory `/etc/OpenCL/vendors` to see if there are any "Installable Client Drivers"
(ICDs) available.  If you see the file `pocl.icd`, then you should at least be able to use a CPU in absence of a
GPU, but you'll notice that simulations are very slow.  Ideally, you'll find vendor-specific ICD files such as
`nvidia.icd` or `intel.icd`.  You need to make sure that these files can be found by the pyopencl module, which probably
means that you need to create a symbolic link like this:

.. code-block:: bash

    ln -s /etc/OpenCL/vendors/intel.icd ~/miniconda3/etc/OpenCL/vendors

If the above is not sufficient, then unfortunately, it is up to you to figure out how to get pyopencl working on your
machine.

Testing your setup
------------------

You can simply move into the test directory and run pytest:

.. code-block:: bash

    cd path/to/reborn/repository
    cd test
    pytest

With some luck, you will get a nice clean output from pytest that looks like the following:

.. code-block:: bash

    ============================= test session starts ==============================
    platform darwin -- Python 3.6.7, pytest-3.9.3, py-1.7.0, pluggy-0.8.0
    rootdir: /Users/rkirian/work/projects/reborn/test, inifile:collected 36 items

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
