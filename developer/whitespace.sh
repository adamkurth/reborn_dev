#!/bin/bash

for file in $(find ../bornagain -name '*.py'); do
perl -pi -e 's/ +$//' $file
#autopep8 -i $file
done

