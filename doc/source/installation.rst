Installation
============

In principle, setting up reborn should be as simple as including the base directory of the reborn git repository in
your python path.  You'll probably find that you need to install some dependencies, but most of them are known to
install with pip or conda.  There may be some extra steps to enable GPU computations, as discussed below.  There are
some pieces of Fortran code that occasionally need to be compiled manually, also discussed below.

Since reborn's interface is not considered "stable", the recommended installation procedure consists of keeping a
clone of the reborn git repository where you are doing your analysis or simulations so that you can reproduce your
results in the future.  You should be mindful of the exact git commit that you use in your analysis, since things may
change in a month from now.  One way to track the *exact* version of reborn used in your project is to add it as a
`git submodule <https://git-scm.com/book/en/v2/Git-Tools-Submodules>`_ to your project's git repository.

Getting reborn
--------------

The first thing you will need to do is use git to clone reborn on your computer:

.. code-block:: bash

    git clone https://gitlab.com/kirianlab/reborn.git

Dependencies
------------

As of 2020, reborn is only tested with Python 3, because
`Python 2 is finally dead <https://www.python.org/doc/sunset-python-2/>`_.
The `environment.yml` file lists all of the packages that are installed upon regular testing of reborn, which might be
understood as the complete list of dependencies.  Here are the current contents of that file:

.. literalinclude:: ../../environment.yml

You can import many reborn modules without installing *all* of these dependencies, but if you use a good package
manager there is little reason to avoid installing all of them.

Setting up Python with Miniconda
--------------------------------

|Miniconda| is a reliable and lightweight distribution of python that is known to
work well with reborn.  The `Conda <https://conda.io/docs/>`_ package manager that comes with it makes it fast and
easy to install and maintain the dependencies of reborn.  It is recommended that you first make a trial
`conda environment <https://conda.io/docs/user-guide/tasks/manage-environments.html>`_ to check that there are no
conflicts between dependencies.  The simplest way to setup a conda environment with all the needed reborn dependencies
is to execute the following command in the *base directory* of the reborn git repository:

.. code-block:: bash

    conda env create --name reborn --file environment.yml

The above line will create the conda environment named ``reborn`` (you may choose a different name if you wish).  You will
need to *activate* that environment whenever you want to use it:

.. code-block:: bash

    source activate reborn

If you wish, you can also install reborn into the default ``base`` conda environment (or another environment that
already exists):

.. code-block:: bash

    conda env update --name base --file environment.yml

Installing reborn with pip
--------------------------

The recommended way to install reborn is to use pip as follows:

.. code-block:: bash

    pip install --no-deps --editable .

You should execute the above *from the base directory* of the git repository.  The ``--editable`` flag in the above
means that you usually do not need to reinstall when you pull the latest updates the reborn git repository.  However,
it may be necessary to reinstall if Fortran code has changed.

Installing reborn without pip
-----------------------------

If you do not want to install with pip, you can simply add the reborn repository to the python search path.  This can be
done by setting the appropriate environment variable:

.. code-block:: bash

    export PYTHONPATH=$PYTHONPATH:example/path/to/reborn/repository

It might be convenient to add the above line to your `bash startup script
<https://www.gnu.org/software/bash/manual/html_node/Bash-Startup-Files.html>`_.

Compilation of Fortran code
---------------------------

We use the f2py program that comes with Numpy to compile Fortran code.  Fortran code should auto-compile for most
users, but in some circumstances you may wish to compile manually.  This can be done using the ``setup.py`` script as
follows:

.. code-block:: bash

    python setup.py develop

If the above fails, you can have a look at the ``developer/compile-fortran.sh`` script.  There are occasionally issues
caused by the mixing of dynamic libraries between different Python versions (not our fault...).

Setting up OpenCL for GPU computing
-----------------------------------

In some cases, |pyopencl| installs via conda without the need for any further modifications.  However, if you have
runtime errors, it may be necessary to install drivers and/or developer toolkits that are specific to your GPU and OS.
You should read through the tips in the official `documentation <https://documen.tician.de/pyopencl/misc.html>`_.  Most
often, you need to look in the directory ``/etc/OpenCL/vendors`` to see if there are any
"Installable Client Drivers" (ICDs) available.  If you see the file ``pocl.icd``, then you should at least be able to
use a CPU as a substitute for a GPU, but you will surely notice that simulations are very slow.  Ideally, you will find
vendor-specific ICD files such as ``nvidia.icd`` or ``intel.icd``.  You need to make sure that these files can be found
by the |pyopencl| module, which *probably* means that you need to create a symbolic link like this:

.. code-block:: bash

    ln -s /etc/OpenCL/vendors/intel.icd ~/miniconda3/etc/OpenCL/vendors

If the above fails, it is unfortunately up to you to figure out how to get |pyopencl| working on your machine.

Testing your setup
------------------

You can simply move into the test directory and run |pytest|:

.. code-block:: bash

    cd path/to/reborn/repository
    cd test
    pytest

With some luck, you will get a nice clean output that looks like the following:

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

Linux notes
-----------

Below are the standard steps that are tested on a clean Ubuntu Linux install every time the reborn master branch is
updated.  The script ``ubuntu_setup.sh`` is executed from the base directory of the git repository.  Here
are the current contents of that script:

.. literalinclude:: ../../ubuntu_setup.sh

Mac OS notes
------------

On Mac OS you need to follow the relevant instructions to install |Miniconda|.  The subsequent steps are *usually*
identical to those you would follow for a Linux setup.  If you do not want to use |Miniconda|, then you will need to
figure out how to get all the dependencies installed.  It might help to install xcode, Homebrew, or similar, to get
gfortran, for example.

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

then please read the relevant section conerning GPUs above.

**Intel GPUs on Ubuntu**

If you have an Ubuntu-like OS and a laptop with an intel GPU you might find
`this github page <https://github.com/intel/compute-runtime/releases>`_ helpful.  After following the instructions
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
