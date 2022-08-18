#!/bin/bash
# You Only Run Once!

conda env create -f deepopt.yml

cd torch-radon
git checkout v2
git pull
python make.py --local
python setup.py install

conda activate deepopt
echo >> 'Environment created succesfully'

