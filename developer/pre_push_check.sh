#!/bin/bash

if [[ ! $(basename "$(pwd)") = 'developer' ]]; then
  echo 'This script should run in the developer directory.'
  exit
fi

bash cleanup_everything.sh
bash pip_install.sh
bash run_tests.sh
bash update_docs.sh
