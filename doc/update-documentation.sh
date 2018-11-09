#!/bin/bash


if [ "$1" = "remake" ]; then 
	make html
	cp -r build/html/* html
	exit
fi

# Remove files auto-generated by sphinx-apidoc
rm source/bornagain.*rst
#rm source/modules.rst
# Note that sphinx-apidoc exclude patterns need an absolute path.  This program sucks.  The documentation is worse.
[ "$1" = "noapi" ] || sphinx-apidoc -o source -e -M ../bornagain ../bornagain/scatter*
make clean
make html
cp -r build/html/* html
rm source/bornagain*.rst # modules.rst
