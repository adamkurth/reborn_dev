#!/usr/bin/env bash

# Opening files from the command line is easy with Mac OS.  How do you do this in Linux?
if [ "`echo $(uname -a | grep Darwin)`" != '' ]; then
 open ../doc/html/index.html
 exit
fi

echo "You need to manually open the docs in ../doc/html/index.html"