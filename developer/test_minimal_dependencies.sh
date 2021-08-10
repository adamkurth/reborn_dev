#!/bin/bash

if [[ "$(conda env list | grep reborn-minimal)" == "" ]]; then
  conda create --name reborn-minimal python=3.7 scipy
fi

source activate reborn-minimal
#./build_inplace.sh
export PYTHONPATH=".."
python << EOF
from reborn import source, detector, utils
p = detector.PADGeometry()
assert p is not None
b = source.Beam()
assert b is not None
assert utils is not None
EOF
