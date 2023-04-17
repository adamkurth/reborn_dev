#!/bin/bash
# Print a list of runs that are available on disk
ls /reg/d/psdm/mfx/mfxp17218/xtc | grep s00-c00 | cut -d '-' -f 2 | sed 's/r//g'

