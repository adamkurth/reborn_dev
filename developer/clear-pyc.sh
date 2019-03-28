#!/bin/bash

cd ..
find . -name '*.pyc' -type f -exec rm {} \+
find . -name '*cache*' -exec rm {} \+
