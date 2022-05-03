Research Computing 
=======================

This is a collection of notes that applies to using reborn on research computing facilities.

On ASU Agave, the command below requests an interactive GPU node: 
.. code-block:: bash

   interactive -p gpu -q wildfire -t 60 --gres=gpu:1 

-t 60 gives you 60 minutes
-gres specifies the number of GPUs to request 
-p means partition
-q the type of queue

For more information, see the webpage below
https://asurc.atlassian.net/wiki/spaces/RC/pages/45678646/Using+Graphics+Processing+Units+GPUs


