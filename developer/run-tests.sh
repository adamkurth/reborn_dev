#!/bin/bash

./compile-fortran.sh
cd ../test
pytest #py.test -p no:cacheprovider -m 'not gui'
