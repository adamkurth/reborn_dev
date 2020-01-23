#!/bin/bash

cd ..
rm -r build
python setup.py develop
rm -r build