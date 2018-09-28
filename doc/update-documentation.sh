#!/bin/bash


if [ "$1" = "remake" ]; then 
	make html
	cp -r build/html/* html
	exit
fi

# Remove files auto-generated by sphinx-apidoc
rm source/bornagain.*rst
#rm source/modules.rst
[ "$1" = "noapi" ] || sphinx-apidoc -M -e -o source ../bornagain
make clean
make html
cp -r build/html/* html
rm source/bornagain*.rst # modules.rst
