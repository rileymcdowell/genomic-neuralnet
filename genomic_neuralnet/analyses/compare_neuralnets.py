from __future__ import print_function
import numpy as np

from functools import partial
from genomic_neuralnet.common import run_predictors
from genomic_neuralnet.methods import \
        get_nn_prediction , get_rbf_nn_prediction, \
        get_fast_nn_prediction, get_fast_nn_dom_prediction, \
        get_cascade_nn_prediction

optimal_rbf_nn = partial(get_rbf_nn_prediction, hidden=(2,), weight_decay=0)
optimal_fast_nn = partial(get_fast_nn_prediction, hidden=(2,), weight_decay=0)
optimal_fast_nn_dom = partial(get_fast_nn_dom_prediction, hidden=(2,), weight_decay=0)

prediction_functions = [  optimal_fast_nn,  optimal_rbf_nn,  optimal_fast_nn_dom ]
prediction_names     = [ 'normal_nn'     , 'rbf_nn'       , 'dominance_nn'       ] 

def main():
    accuracies = run_predictors(prediction_functions)

    print('')
    for name, accuracy_arr in zip(prediction_names, accuracies):
        print('{} accuracy: mean {} sd {}'.format(name, np.mean(accuracy_arr), np.std(accuracy_arr)))
    print('Done')

if __name__ == '__main__':
    main()

