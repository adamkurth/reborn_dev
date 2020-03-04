#!/usr/bin/env bash

htmlindex="../doc/html/index.html"

# Opening files from the command line is easy with Mac OS.  How do you do this in Linux?
if [ "`echo $(uname -a | grep Darwin)`" != '' ]; then
 open ${htmlindex}
 exit
elif [ $(command -v xdg-open) ]; then
 xdg-open ${htmlindex}
 exit
fi

echo "You need to manually open the docs in ../doc/html/index.html"