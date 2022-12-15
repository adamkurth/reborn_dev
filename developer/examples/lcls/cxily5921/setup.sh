# Setup python 3
source /reg/g/psdm/etc/psconda.sh -py3  # load analysis python environment
export PS1='ly59> '  # convenient way to know we are in an analysis environment
export PYTHONPATH=$(cd ../../../../../reborn; pwd)  # tell python where to find reborn