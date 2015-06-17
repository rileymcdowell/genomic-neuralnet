from __future__ import print_function
import numpy as np

from functools import partial
from genomic_neuralnet.common import run_predictors
from genomic_neuralnet.methods import \
        get_nn_prediction , get_rbf_nn_prediction

optimal_nn = partial(get_nn_prediction, hidden=(2,), weight_decay=0.025) # Optimal hidden layer neurons == 2
# TODO: Get this to return sane values 
optimal_rbf_nn = partial(get_rbf_nn_prediction, hidden=(2,), weight_decay=0.025)
prediction_functions = [  optimal_nn    ,  optimal_rbf_nn ] 
prediction_names     = [ 'neuralnet'    , 'rbf_neuralnet' ] 

def main():
    accuracies = run_predictors(prediction_functions)

    print('')
    for name, accuracy_arr in zip(prediction_names, accuracies):
        print('{} accuracy: mean {} sd {}'.format(name, np.mean(accuracy_arr), np.std(accuracy_arr)))
    print('Done')

if __name__ == '__main__':
    main()

