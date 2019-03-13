#!/bin/bash

find .. -name '*.so'        -type f -exec rm {} \+
find .. -name '*.dSYM'      -type d -exec rm -r {} \+
find .. -name '*.pyc'       -type f -exec rm {} \+
find .. -name '__pycache__' -type f -exec rm -r {} \+
