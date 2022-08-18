#!/bin/bash
# You Only Run Once!

find_in_conda_env(){
    conda env list | grep "${@}" >/dev/null 2>/dev/null
}

if find_in_conda_env ".*deepopt.*" ; then
   conda activate deepopt
   conda env update --file deepopt.yml --prune
else 
    conda env create -f deepopt.yml
fi

cd torch-radon
git checkout v2
git pull
python make.py --local
python setup.py install

conda activate deepopt
echo >> 'Environment created succesfully'

