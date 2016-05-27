#!/bin/bash
source $(which virtualenvwrapper.sh) 

# Activate your virtual environment.
VIRTUALENV_NAME=genomic_sel
workon ${VIRTUALENV_NAME} 

# Add CUDA to path.
if [ -d "/usr/local/cuda/bin" ] ; then
    PATH="$PATH:/usr/local/cuda/bin"
    LD_LIBRARY_PATH=/usr/local/cuda/lib64/:/usr/lib/nvidia-313-updates/
fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

# This is designed to be called by a cron-job like follows:
# * * * * * $HOME/{path_to_this_repo}/start_tensorboard.sh 
# Your crontab doesn't know the path to this repo, so you need to fill that in yourself.

# Use flock to make this a service.
REPO_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
flock $REPO_DIR/tf_log_dir/tensorboard.lockfile tensorboard --logdir ${REPO_DIR}/tf_log_dir
