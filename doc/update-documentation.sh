#!/bin/bash

# cd $(dirname $0)

# Note: when documentation is created and pushed to the gitlab repository, this script will run automatically and
# documentation will be made available here:
#
# https://rkirian.gitlab.io/bornagain

rm -r source/api
sphinx-apidoc --output-dir source/api --module-first ../bornagain
ls source/api
# Fix the stupid default title of API page
tail -n+3 source/api/modules.rst > tmp.rst
echo 'API Reference' > source/api/modules.rst
echo '=============' >> source/api/modules.rst
cat tmp.rst >> source/api/modules.rst

rm tmp.rst
make clean
make doctest
make html
rm -r html
mv build/html .
perl -p -i -e 's{<head>\n}{<head>\n  <meta name="robots" content="noindex, nofollow" />\n}' html/*.html
