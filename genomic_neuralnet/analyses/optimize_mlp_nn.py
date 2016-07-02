from __future__ import print_function

from genomic_neuralnet.config import SINGLE_CORE_BACKEND, JOBLIB_BACKEND
from genomic_neuralnet.methods import get_net_prediction
from genomic_neuralnet.analyses import run_optimization

from genomic_neuralnet.util import get_is_on_gpu

from sklearn.utils.extmath import cartesian

def main():
    hidden_sizes = (1, 2, 4, 8) #
    cartesian([hidden_sizes])
    one_layer = cartesian((hidden_sizes,)*1) 
    two_layers = cartesian((hidden_sizes,)*2) 
    three_layers = cartesian((hidden_sizes,)*3) 

    tup = lambda x: map(tuple, x)
    hidden = tup(one_layer) + tup(two_layers) + tup(three_layers)

    params = { 'hidden': hidden 
             , 'batch_size': (100,)
             , 'epochs'    : (1000,)
             }

    backend = JOBLIB_BACKEND
    if get_is_on_gpu():
        backend = SINGLE_CORE_BACKEND

    run_optimization(get_net_prediction, params, 'optimal_nn.shelf', 'N', backend=backend)

if __name__ == '__main__':
    main()

