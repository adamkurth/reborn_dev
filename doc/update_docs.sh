#!/bin/bash

# cd $(dirname $0)

# Note: when documentation is created and pushed to the gitlab repository, this script will run automatically and
# documentation will be made available here:
#
# https://rkirian.gitlab.io/reborn

#[ -d source/api ] && rm -r source/api
#[ -d source/auto_examples ] && rm -r source/auto_examples
sphinx-apidoc --output-dir source/api --module-first ../reborn ../reborn/fortran ../reborn/simulate/atoms.py ../reborn/simulate/numbacore.py
# FIXME: How do we properly change the title of the auto-generated API page?  Below we do it brute-force...
#tail -n+3 source/api/modules.rst > tmp.rst
#echo 'API Reference' > source/api/modules.rst
#echo '=============' >> source/api/modules.rst
#cat tmp.rst >> source/api/modules.rst
#rm tmp.rst &> /dev/null
make clean
make doctest
make html
#perl -p -i -e 's{<head>\n}{<head>\n  <meta name="robots" content="noindex, nofollow" />\n}' build/html/*.html
#perl -p -i -e 's{toctree-l2}{toctree-l1}' build/html/api/modules.html
#perl -p -i -e 's{toctree-l3}{toctree-l1}' build/html/api/modules.html
#perl -p -i -e 's{toctree-l4}{toctree-l1}' build/html/api/modules.html
#perl -p -i -e 's{>reborn.*</a>}{}'