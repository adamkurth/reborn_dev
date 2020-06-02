#!/bin/bash 

# Automatically corrects simple violations of PEP8.  Make sure you check what changes were made...

autopep8 --aggressive --max-line-length 120 $1 # --in-place

