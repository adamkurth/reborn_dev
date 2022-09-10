#!/bin/bash
# Checks for compliance with PEP8, excluding some exceptions.
[ "$1" = "" ] && exit
sed --in-place 's/[[:space:]]\+$//' "$1"
pycodestyle --first --max-line-length 120 --show-source "$1"
pylint --disable too-few-public-methods \
       --disable too-many-instance-attributes \
       --disable too-many-arguments \
       --disable invalid-name \
		   --disable too-many-public-methods \
		   --disable too-many-lines \
		   --disable too-many-return-statements \
		   --disable too-many-locals \
		   --max-line-length=120 \
       --extension-pkg-whitelist=numpy,reborn.fortran \
       $1
