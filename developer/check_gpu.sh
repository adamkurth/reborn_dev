#!/bin/bash

python << EOF
from reborn.simulate import clcore
print(clcore.__file__)
clcore.help()
EOF
