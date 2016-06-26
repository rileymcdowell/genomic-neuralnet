from __future__ import print_function

from genomic_neuralnet.config import SINGLE_CORE_BACKEND, JOBLIB_BACKEND
from genomic_neuralnet.analyses import run_optimization
from genomic_neuralnet.methods import get_net_prediction
from genomic_neuralnet.util import get_is_on_gpu

def main():
    hidden_size = map(lambda x: tuple([x]), (1, 2, 4, 8, 16, 32, 64, 128, 256, 512))
    dropout = (0.1, 0.2, 0.3, 0.4, 0.5)
    params = { 'hidden': hidden_size
             , 'dropout_prob': dropout
             , 'batch_size': (100,)
             , 'epochs': (1000,)
             }

    backend = JOBLIB_BACKEND
    if get_is_on_gpu():
        backend = SINGLE_CORE_BACKEND

    run_optimization(get_net_prediction, params, 'optimal_donn.shelf', backend=backend)

if __name__ == '__main__':
    main()

