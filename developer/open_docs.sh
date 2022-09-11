#!/usr/bin/env bash
if [[ ! $(basename "$(pwd)") = 'developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit
fi
cd ..
html_index="$(pwd)/doc/build/html/index.html"
# Opening files from the command line is easy with Mac OS.  How do you do this in Linux?
if [ "$(uname -a | grep Darwin)" != '' ]; then
 open "${html_index}"
 exit
elif [ "$(command -v sensible-browser)" ]; then
 sensible-browser "${html_index}"
 exit
elif [ "$(command -v xdg-open)" ]; then
 xdg-open "${html_index}"
 exit
fi
echo "You need to manually open the docs in ../doc/build/html/index.html"
