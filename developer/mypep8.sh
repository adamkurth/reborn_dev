#!/bin/bash 

# Automatically corrects simple violations of PEP8.  Make sure you check what changes were made...

pep8 --max-line-length 120 $1
