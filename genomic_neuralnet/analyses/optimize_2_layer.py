from __future__ import print_function
import sys
import numpy as np
import scipy as sp
import pandas as pd
from joblib import delayed, Parallel, cpu_count
from functools import partial

from common import run_predictors
from methods import get_nn_prediction


# 2 hidden layers, try all combinations less than 5x5.
hidden_layers = [(x,y) for x in range(1,5) for y in range(1,5)]
prediction_functions = [partial(get_nn_prediction, hidden=h) for h in hidden_layers]
prediction_names = ['2 layer nn {}'.format(h) for h in hidden_layers]

def main():
    accuracies = run_predictors(prediction_functions)

    print('')
    for name, accuracy_arr in zip(prediction_names, accuracies):
        print('{} accuracy: mean {} sd {}'.format(name, np.mean(accuracy_arr), np.std(accuracy_arr)))
    print('Done')

if __name__ == '__main__':
    main()

