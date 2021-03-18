#!/bin/bash

if [[ ! $(basename "$(pwd)") = 'developer' ]]; then
    echo 'This script should run in the developer directory.'
    exit 1
fi

#bash cleanup_everything.sh
#bash pip_install.sh
#echo '============= importing reborn.fortran ============='
#python -c 'import reborn.fortran'
#echo '============= done ================================='
#echo '============= importing reborn.fortran ============='
#python -c 'import reborn.fortran'
#echo '============= done ================================='
#pip uninstall reborn

bash cleanup_everything.sh
bash compile_fortran.sh
cd ..
echo '========= importing reborn.fortran without compiling ======='
echo '============= (does it compile automatically?) ============='
python -c 'import reborn.fortran'

echo '=========== importing reborn.fortran again ================='
echo '======= (did it need to re-compile?  it should not) ========'
python -c 'import reborn.fortran; reborn.fortran.omp_test_f.omp_test()'
echo "$(python -c 'import reborn.fortran; reborn.fortran.omp_test_f.omp_test()' | wc -l) omp cores"

cd developer || exit
bash cleanup_everything.sh
echo '========== running build_inplace.sh ======================='
bash build_inplace.sh
cd ..
echo '============= importing reborn.fortran ============='
python -c 'import reborn.fortran'
echo '============= importing reborn.fortran again (should not re-compile) ============='
python -c 'import reborn.fortran; reborn.fortran.omp_test_f.omp_test()'
echo "$(python -c 'import reborn.fortran; reborn.fortran.omp_test_f.omp_test()' | wc -l) omp cores"
cd developer || exit
bash cleanup_everything.sh
echo '============= done ================================='
