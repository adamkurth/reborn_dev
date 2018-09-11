#!/bin/bash

find . -type f -name '*.py' -print 0 | xargs -0 perl -pi -e 's/ +$//'
