from __future__ import print_function

from genomic_neuralnet.config import SINGLE_CORE_BACKEND, JOBLIB_BACKEND
from genomic_neuralnet.methods import get_net_prediction
from genomic_neuralnet.analyses import run_optimization, HIDDEN, EPOCHS, BATCH_SIZE

from genomic_neuralnet.util import get_is_on_gpu

from sklearn.utils.extmath import cartesian

def main():
    params = { 'hidden': HIDDEN 
             , 'batch_size': (BATCH_SIZE,)
             , 'epochs'    : (EPOCHS,)
             }

    backend = JOBLIB_BACKEND
    if get_is_on_gpu():
        backend = SINGLE_CORE_BACKEND

    run_optimization(get_net_prediction, params, 'optimal_nn.shelf', 'N', backend=backend)

if __name__ == '__main__':
    main()

