from __future__ import print_function

from genomic_neuralnet.config import SINGLE_CORE_BACKEND, JOBLIB_BACKEND
from genomic_neuralnet.methods import get_net_prediction
from genomic_neuralnet.analyses import run_optimization, HIDDEN, DROPOUT, WEIGHT_DECAY, BATCH_SIZE, EPOCHS

from genomic_neuralnet.util import get_is_on_gpu, get_is_time_stats, get_should_plot

def main():
    params = { 'hidden': HIDDEN 
             , 'dropout_prob': DROPOUT 
             , 'weight_decay': WEIGHT_DECAY 
             , 'epochs': (EPOCHS,)
             }

    backend = JOBLIB_BACKEND
    if get_is_on_gpu() or get_is_time_stats() or get_should_plot():
        backend = SINGLE_CORE_BACKEND

    run_optimization(get_net_prediction, params, 'optimal_wddonn.shelf', 'NWDDO', backend=backend, retry_nans=True)

if __name__ == '__main__':
    main()

