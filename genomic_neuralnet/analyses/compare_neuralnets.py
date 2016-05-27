from __future__ import print_function
import numpy as np

import warnings
warnings.filterwarnings('error', category=DeprecationWarning)

from functools import partial
from genomic_neuralnet.config import SINGLE_CORE_BACKEND 
from genomic_neuralnet.util import NeuralnetConfig
from genomic_neuralnet.common import run_predictors
from genomic_neuralnet.methods import \
        get_nn_prediction , get_rbf_nn_prediction, \
        get_fast_nn_prediction, get_fast_nn_dom_prediction, \
        get_do_net_prediction, get_tfnet_pred

CYCLES = 20

#optimal_rbf_nn = partial(get_rbf_nn_prediction, centers=200, spread=150)
optimal_fast_nn = partial(get_fast_nn_prediction, hidden=(2,), weight_decay=0.0)
#optimal_fast_nn_dom = partial(get_fast_nn_dom_prediction, hidden=(2,), weight_decay=0)
tfnet = partial(get_tfnet_pred, hidden=(2,), weight_decay=0.0)

#do_net_config = NeuralnetConfig()
#do_net_config.batch_splits = 2
#do_net_config.max_epochs = 1000
#do_net = partial(get_do_net_prediction, net_config = do_net_config, dropout_keep_prob=0.75)

#prediction_functions = [  optimal_fast_nn,  optimal_rbf_nn,  optimal_fast_nn_dom,  do_net ]
#prediction_names     = [ 'normal_nn'     , 'rbf_nn'       , 'dominance_nn'      , 'do_net'] 

prediction_functions = [  optimal_fast_nn,  tfnet  ] 
prediction_names     = [ 'normal_nn'     , 'tfnet' ]

def main():
    accuracies = run_predictors(prediction_functions, backend=SINGLE_CORE_BACKEND, cycles=2)

    print('')
    for name, accuracy_arr in zip(prediction_names, accuracies):
        print('{} accuracy: mean {} sd {}'.format(name, np.mean(accuracy_arr), np.std(accuracy_arr)))
    print('Done')

if __name__ == '__main__':
    main()

