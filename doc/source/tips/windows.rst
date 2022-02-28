.. _windows_anchor:

Working with Windows 10
=======================

The best option when working with Windows in general is probably to install a virtual machine with a Linux OS.

Another option for Windows 10 is to install Microsoft's Ubuntu subsystem.  Below are some instructions that
*might* work for you...

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

    export DISPLAY=localhost:0.0

You might want to add the above to your startup script, for example like this:

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
