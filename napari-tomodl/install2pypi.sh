# run this script in the root directory of the project
# ./install2pypi.sh
# This script will remove the build, dist, and *.egg-info directories, create a new distribution, and upload it to PyPI.
# It assumes you have the necessary permissions to upload to PyPI.
rm -rf build dist *.egg-info
python setup.py sdist bdist_wheel
twine upload dist/*
