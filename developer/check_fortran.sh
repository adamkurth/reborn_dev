#!/bin/bash

if [[ ! $(basename "$(pwd)") = 'developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit 1
fi


pip uninstall reborn
bash cleanup_everything.sh
bash pip_install.sh
echo '============= importing reborn.fortran ============='
python -c 'import reborn.fortran'
echo '============= done ================================='
echo '============= importing reborn.fortran ============='
python -c 'import reborn.fortran'
echo '============= done ================================='
pip uninstall reborn

bash cleanup_everything.sh
bash compile_fortran.sh
cd ..
echo '============= importing reborn.fortran ============='
python -c 'import reborn.fortran'
echo '============= done ================================='
echo '============= importing reborn.fortran ============='
python -c 'import reborn.fortran'
echo '============= done ================================='
cd developer

bash cleanup_everything.sh
bash build_inplace.sh
cd ..
echo '============= importing reborn.fortran ============='
python -c 'import reborn.fortran'
echo '============= done ================================='
echo '============= importing reborn.fortran ============='
python -c 'import reborn.fortran'
echo '============= done ================================='
cd developer
bash cleanup_everything.sh

