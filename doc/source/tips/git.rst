.. _gittips_anchor:

Git tips
========

How to add reborn as a submodule
--------------------------------

By adding reborn as a git submodule to your repository, you will be able to track which reborn branch/commit you are
using in your analysis.  You first need to do this:

.. code-block:: bash

    git submodule add -b develop https://gitlab.com/kirianlab/reborn.git

In order to update reborn to the latest commit, do this:

.. code-block:: bash

    #!/bin/bash
    git submodule update --init --recursive --remote

It is probably to put this into a shell script.

Importantly, you should never modify anything within the reborn directory within your project.  You should only
update reborn with the above command.

Once reborn is a submodule, you can automatically fetch it upon cloning your repository by adding the appropriate
flag:

.. code-block:: bash

    git clone --recurse-submodules yourpackage
