#!/bin/bash

sphinx-apidoc -o source -M ../bornagain
make clean
make html
rm -r html
rm source/bornagain*.rst
mv build/html .
perl -p -i -e 's{<head>\n}{<head>\n  <meta name="robots" content="noindex, nofollow" />\n}' html/*.html
