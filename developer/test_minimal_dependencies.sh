#!/bin/bash

if [[ "$(conda env list | grep reborn-minimal)" == "" ]]; then
  conda create --name reborn-minimal python=3.7 scipy
fi

source activate reborn-minimal
#./build_inplace.sh
export PYTHONPATH=".."
python << EOF
import reborn
assert reborn is not None
from reborn import detector
p = detector.PADGeometry()
assert p is not None
from reborn import source
b = source.Beam()
assert b is not None
from reborn import utils
assert utils is not None
EOF
