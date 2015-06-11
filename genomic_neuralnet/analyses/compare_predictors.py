from __future__ import print_function
import numpy as np

from functools import partial
from genomic_neuralnet.common import run_predictors
from genomic_neuralnet.methods import \
        get_brr_prediction, get_en_prediction, \
        get_lasso_prediction, get_lr_prediction, \
        get_nn_prediction, get_rr_prediction

optimal_nn = partial(get_nn_prediction, hidden=(2,), weight_decay=0.025) # Optimal hidden layer neurons == 2
prediction_functions = [optimal_nn     , get_rr_prediction, get_lr_prediction, get_brr_prediction   , get_lasso_prediction, get_en_prediction]
prediction_names     = ['neuralnet'    , 'ridge_reg'    , 'linear_reg'   , 'baesian_ridge_reg', 'lasso_reg'       , 'elastic_net'  ]

def main():
    accuracies = run_predictors(prediction_functions)

    print('')
    for name, accuracy_arr in zip(prediction_names, accuracies):
        print('{} accuracy: mean {} sd {}'.format(name, np.mean(accuracy_arr), np.std(accuracy_arr)))
    print('Done')

if __name__ == '__main__':
    main()

