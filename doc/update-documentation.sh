#!/bin/bash

rm -r source/api
sphinx-apidoc --output-dir source/api --module-first ../bornagain
ls source/api
# Fix stupid title of API page
tail -n+3 source/api/modules.rst > tmp.rst
echo 'API Reference' > source/api/modules.rst
echo '=============' >> source/api/modules.rst
cat tmp.rst >> source/api/modules.rst

rm tmp.rst
make clean
make html
rm -r html
mv build/html .
perl -p -i -e 's{<head>\n}{<head>\n  <meta name="robots" content="noindex, nofollow" />\n}' html/*.html
