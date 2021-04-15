#!/bin/bash

# Some modifications:
# C0103: variables with less than three characters
# R0903: classes with too few methods
# R0205: useless object inheritance 

pylint --disable too-few-public-methods \
       --disable too-many-instance-attributes \
       --disable too-many-arguments \
       --disable invalid-name \
       --max-line-length=120 \
       --extension-pkg-whitelist=numpy,reborn.fortran \
       $1
