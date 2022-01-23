#!/bin/bash
remote=psexport:/reg/d/psdm/mfx/mfxlv1218/scratch/rkirian/reborn_repo
local=../../../../../../reborn/
rsync -arv --stats --progress --exclude='*.md5' --exclude='*.so' --exclude=__pycache__ --exclude='*swp' --exclude=.git $local $remote
#
#remote=ssh://psexport//cds/home/r/rkirian/cxil2316-new
#local=../../../../../../reborn
#force="-force $local"
#unison -ignore='Name {miniconda,reborn.egg-info,*.swp,*.log,*.so,*.md5,build,*.pyc,*cache*,conda_env,anaconda3,results,home}' $1 \
#       -servercmd '~/work/local/bin/unison' $force $local $remote