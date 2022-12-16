#!/bin/bash
ls /reg/d/psdm/cxi/cxily5921/xtc | grep s00-c00 | cut -d '-' -f 2 | sed 's/r//g'

