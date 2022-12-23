#!/bin/bash

USER="$1"
if [ "$USER" == "" ]; then
       echo 'Please include your psana username'
       exit
fi

rsync -arv --exclude doc --exclude tests --exclude '*.so' --exclude '.git' --exclude '*.pyc' --exclude '__pycache__' --exclude '.idea' --exclude '*.md5' ../../../../../reborn "$USER"@psexport.slac.stanford.edu:/reg/d/psdm/cxi/cxily5921/scratch/"$USER"
