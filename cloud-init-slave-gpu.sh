#!/bin/bash

###################################################
# This file sets up a fresh amazon linux box to run 
# the code in this library.
###################################################

USER_NAME=ec2-user
USER_HOME=/home/$USER_NAME

# Start out by getting up-to-date.
yum update -y

# Install system-level dependencies.
yum groupinstall 'Development Tools' -y
yum install freetype-devel libpng-devel -y
yum install cmake -y
yum install blas-devel lapack-devel -y
yum install htop -y

# Set up GPU.
pushd /root # Install into root's home.
yum install libX11-devel -y
yum install kernel-devel-$(uname -r) -y
wget 'http://us.download.nvidia.com/XFree86/Linux-x86_64/367.44/NVIDIA-Linux-x86_64-367.44.run' --quiet
/bin/bash NVIDIA-Linux-x86_64-367.44.run --silent --disable-nouveau --run-nvidia-xconfig
# Set up nvcc.
wget 'http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda_7.5.18_linux.run' --quiet
/bin/bash cuda_7.5.18_linux.run --toolkit --no-opengl-libs --silent
# Set up cudnn.
aws s3 cp s3://instance-cache/dependencies/cudnn-7.0-linux-x64-v4.0-prod.tgz .
tar xvzf cudnn-7.0-linux-x64-v4.0-prod.tgz
cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/
cp cuda/include/cudnn.h /usr/local/cuda/include/
popd

# Easy python dependencies.
pip install pytest mock nose six parse boto3
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

################################
# Install source to user's home. 
################################

# Install simplennet dependency.
pushd $USER_HOME 
sudo -u $USER_NAME git clone https://github.com/rileymcdowell/simplennet.git
pushd simplennet/
#pip install -r requirements.txt # No reqs file...
python setup.py develop
popd

# Install this library.
sudo -u $USER_NAME git clone https://github.com/rileymcdowell/genomic-neuralnet.git
pushd genomic-neuralnet/
pip install -r requirements.txt
python setup.py develop
popd

#####################################
# Set up cron jobs. 
#####################################
cat << EOF >> $USER_HOME/crontab.init
HOME=/home/ec2-user
SHELL=/bin/bash
PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin
LD_LIBRARY_PATH=/usr/local/cuda/lib64
* * * * * flock -n $USER_HOME/celery_slave.lockfile python $USER_HOME/genomic-neuralnet/genomic_neuralnet/common/celery_slave.py --gpu &>> slave_worker.log
EOF

sudo -u $USER_NAME crontab crontab.init

popd # Leave the user's home.


