#!/bin/bash
# You Only Run Once!

find_in_conda_env(){
    conda env list | grep "${@}" >/dev/null 2>/dev/null
}

conda init bash

if find_in_conda_env ".*deepopt.*" ; then

   conda activate deepopt
   conda env update --file deepopt.yml --prune
else 
    conda env create -f deepopt.yml
fi

wget -qO- https://raw.githubusercontent.com/matteo-ronchetti/torch-radon/v2/auto_install.py | python -

conda activate deepopt
echo >> 'Environment created succesfully'

