#!/bin/bash

# Some modifications:
# C0103: variables with less than three characters
# R0903: classes with too few methods
# R0205: useless object inheritance 

pylint --disable C0103 \
       --disable R0903 \
       --max-line-length=120 $1
