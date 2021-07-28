#!/bin/bash 

# Checks for compliance with PEP8, excluding some exceptions.

pycodestyle --first --max-line-length 120 --show-source $1
