#!/bin/bash

rsync -arv --exclude '*.so' --exclude '.git' --exclude '*.pyc' --exclude '__pycache__' --exclude '.idea' --exclude '*.md5' /home/rkirian/work/projects/reborn psexport:/reg/d/psdm/mfx/mfxp17218/scratch/rkirian
