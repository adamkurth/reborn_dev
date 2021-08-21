#!/bin/bash 

# Checks for compliance with PEP8, excluding some exceptions.

[ "$1" = "" ] && exit

sed --in-place 's/[[:space:]]\+$//' "$1"
pycodestyle --first --max-line-length 120 --show-source "$1"
