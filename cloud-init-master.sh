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

# Install Redis.
wget http://download.redis.io/releases/redis-3.2.3.tar.gz
tar xvzf redis-3.2.3.tar.gz
pushd redis-3.2.3
make -j
make install
echo 'vm.overcommit_memory = 1' >> /etc/sysctl.conf
sysctl vm.overcommit_memory=1
echo 512 > /proc/sys/net/core/somaxconn
echo 'echo 512 > /proc/sys/net/core/somaxconn' >> /etc/rc.local
echo never > /sys/kernel/mm/transparent_hugepage/enabled
echo 'echo never > /sys/kernel/mm/transparent_hugepage/enabled' >> /etc/rc.local
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
PATH=/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin
*/20 * * * * aws s3 sync $USER_HOME/genomic-neuralnet/genomic_neuralnet/analyses/shelves/ s3://instance-cache &>> $USER_HOME/cache-sync.log
* * * * * flock -n $USER_HOME/redis.lockfile redis-server --maxclients 128 --bind 0.0.0.0 &>> $USER_HOME/redis-server.log
EOF

sudo -u $USER_NAME crontab crontab.init

popd # Leave the user's home.


