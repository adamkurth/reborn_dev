#!/bin/bash

cd ../test
py.test -p no:cacheprovider -m 'not gui'
