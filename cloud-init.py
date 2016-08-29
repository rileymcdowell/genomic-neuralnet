#!/bin/bash

###################################################
# This file sets up a fresh amazon linux box to run 
# the code in this library.
###################################################

# Start out by getting up-to-date.
yum update -y

# Install system-level dependencies.
yum groupinstall 'Development Tools' -y
yum install freetype-devel libpng-devel
yum install cmake -y
yum install blas-devel -y
yum install lapack-devel -y

# Easy python dependencies.
pip install pytest mock nose six parse
pip install joblib celery redis
pip install bz2file

# Harder scientific python dependencies.
pip install numpy
pip install scipy
pip install pandas
pip install statsmodels
pip install scikit-learn
pip install Keras
pip install matplotlib
pip install ipython

# Install FANN.
git clone https://github.com/libfann/fann.git
pushd fann
cmake .
make install -j
pip install fann2
popd

# Load libs (like fann2) in /usr/local/lib
pushd /etc/ld.so.conf.d/
echo '/usr/local/lib' > local_lib.conf
ldconfig
popd

# Install simplennet dependency.
git clone https://github.com/rileymcdowell/simplennet.git
pushd simplennet/
#pip install -r requirements.txt
python setup.py develop
popd

# Install this library.
git clone https://github.com/rileymcdowell/genomic-neuralnet.git
pushd genomic-neuralnet/
pip install -r requirements.txt
python setup.py develop
popd

