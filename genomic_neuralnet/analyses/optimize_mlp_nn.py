from __future__ import print_function

from genomic_neuralnet.config import SINGLE_CORE_BACKEND, PARALLEL_BACKEND
from genomic_neuralnet.methods import get_net_prediction
from genomic_neuralnet.analyses import run_optimization, HIDDEN, EPOCHS, \
                                       DOUBLE_MULTIPLIER

from genomic_neuralnet.util import get_is_on_gpu, get_is_time_stats, get_should_plot

from sklearn.utils.extmath import cartesian

def main():
    params = { 'hidden': HIDDEN 
             , 'epochs'    : (EPOCHS,)
             }

    backend = PARALLEL_BACKEND
    if get_is_on_gpu() or get_is_time_stats() or get_should_plot():
        backend = SINGLE_CORE_BACKEND

    run_optimization( get_net_prediction, params, 'optimal_nn.shelf', 'N'
                    , sample_size_multiplier=DOUBLE_MULTIPLIER
                    , backend=backend, retry_nans=True)

if __name__ == '__main__':
    main()

