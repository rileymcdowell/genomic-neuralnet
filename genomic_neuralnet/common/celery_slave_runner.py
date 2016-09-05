from __future__ import absolute_import, print_function

import os
import sys

from celery.bin import worker
from genomic_neuralnet.common.celery_slave import app

# Set configuration for worker.
args = ['worker', '--loglevel=INFO', '-Ofair', '--without-gossip', '--without-mingle']
if '--gpu' in sys.argv:
    print('Configuring worker to use GPU training.')
    os.environ['PATH'] = os.environ.get('PATH', '') + os.pathsep + '/usr/local/cuda/bin'
    os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + os.pathsep + '/usr/local/cuda/lib64'
    os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu,' \
                                 'lib.cnmem=0.85,nvcc.fastmath=True,' \
                                 'mode=FAST_RUN,blas.ldflags="-lblas -llapack",'
    args.extend(['--concurrency', '1'])

def main():
    print('STARTING WORKER')

    # Configure the app.
    app.config_from_object('genomic_neuralnet.common.celeryconfig')

    # Start workers.
    app.worker_main(args) 

if __name__ == '__main__':
    main()

