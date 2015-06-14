from __future__ import print_function
import sys
import numpy as np

from functools import partial

from genomic_neuralnet.common import run_predictors 
from genomic_neuralnet.methods import get_nn_prediction

# These are probably reasonable values. Find the global maximum accuracy.
layer_size = [2]
layer_size = map(lambda x: tuple([x]), layer_size)
decay_size = list(np.arange(0.00, 0.0501, 0.005))

params = [{'hidden': h, 'weight_decay': wd} for h in layer_size for wd in decay_size]
prediction_functions = map(lambda args: partial(get_nn_prediction, **args), params)
prediction_names = tuple(['nn h:{}, wd:{}'.format(p['hidden'], p['weight_decay']) for p in params]) 

def main():
    accuracies = run_predictors(prediction_functions)

    print('')
    for name, accuracy_arr in zip(prediction_names, accuracies):
        print('{} accuracy: mean {} sd {}'.format(name, np.mean(accuracy_arr), np.std(accuracy_arr)))
    for decay, accuracy_arr in zip(decay_size, accuracies):
        print(', ({}, {}, {})'.format(decay, np.mean(accuracy_arr), np.std(accuracy_arr)))
    print('Done')

if __name__ == '__main__':
    main()

