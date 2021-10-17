#!/bin/bash

cd ../developer
bash cleanup_everything.sh
cd ../doc
make clean
bash update_docs.sh
