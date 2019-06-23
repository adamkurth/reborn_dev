#!/bin/bash

cd ..
find .. -name '__pycache__' -type d
find . -name '*.pyc' -type f -exec rm {} \+
find .. -name '.pytest_cache' -type d -exec rm -r {} \;
#find . -name '*cache*' -exec rm {} \+
