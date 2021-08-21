#!/bin/bash

# cd $(dirname $0)

# Note: when documentation is created and pushed to the gitlab repository, this script will run automatically and
# documentation will be made available here:
#
# https://rkirian.gitlab.io/reborn

export PYTHONPATH=$(pwd)/source:$PYTHONPATH  # Needed for qtgallery
pwd
[ -d source/api ] && rm -r source/api
#[ -d source/auto_examples ] && rm -r source/auto_examples
sphinx-apidoc --maxdepth 10 --output-dir source/api ../reborn \
 ../reborn/data ../reborn/fortran ../reborn/simulate/atoms.py ../reborn/simulate/numbacore.py
# FIXME: How do we properly change the title of the auto-generated API page?  Below we do it brute-force...
tail -n+3 source/api/modules.rst > tmp.rst
echo 'reborn API' > source/api/modules.rst
echo '==========' >> source/api/modules.rst
cat tmp.rst >> source/api/modules.rst
rm tmp.rst &> /dev/null
#make clean
make doctest
make html
prev=$(pwd)
cd latex/dipole
make
cd $prev
cp -r source/files build/html
#exit
# Attempting to make the ugly API more readable... failed
#sed -i.bak '/>*package</d' build/html/api/modules.html
#sed -i.bak '/>*contents</d' build/html/api/modules.html
sed -i.bak '/>*Submodules</d' build/html/api/modules.html
#sed -i.bak '/>*Subpackages</d' build/html/api/modules.html
sed -i.bak '/>Module contents</d' build/html/api/modules.html
#sed -i.bak '/>*package</d' build/html/api/modules.html
sed -i.bak 's/ package</</g' build/html/api/modules.html
sed -i.bak 's/ module</</g' build/html/api/modules.html
#sed -i.bak 's/-l3/-l2/g' build/html/api/modules.html
#sed -i.bak 's/-l4/-l3/g' build/html/api/modules.html
#sed -i.bak 's/-l5/-l4/g' build/html/api/modules.html
#sed -i.bak 's/-l6/-l5/g' build/html/api/modules.html

#perl -p -i -e 's{<head>\n}{<head>\n  <meta name="robots" content="noindex, nofollow" />\n}' build/html/*.html
#perl -p -i -e 's{toctree-l2}{toctree-l1}' build/html/api/modules.html
#perl -p -i -e 's{toctree-l3}{toctree-l1}' build/html/api/modules.html
#perl -p -i -e 's{toctree-l4}{toctree-l1}' build/html/api/modules.html
#perl -p -i -e 's{>reborn.*</a>}{}'
