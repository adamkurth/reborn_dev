#!/bin/bash
ls /reg/d/psdm/amo/amox26916/xtc | grep s00-c00 | cut -d '-' -f 2 | sed 's/r//g'
