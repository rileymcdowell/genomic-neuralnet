from __future__ import print_function
import sys
import numpy as np

from functools import partial

from common import run_predictors 
from methods import get_nn_prediction

prediction_functions = [ partial(get_nn_prediction, hidden=(1,))
                       , partial(get_nn_prediction, hidden=(2,))
                       , partial(get_nn_prediction, hidden=(3,))
                       , partial(get_nn_prediction, hidden=(4,))
                       ]
prediction_names = tuple(['nn {}'.format(2**x) for x in range(1, len(prediction_functions) + 1)])

def main():
    accuracies = run_predictors(prediction_functions)

    print('')
    for name, accuracy_arr in zip(prediction_names, accuracies):
        print('{} accuracy: mean {} sd {}'.format(name, np.mean(accuracy_arr), np.std(accuracy_arr)))
    print('Done')

if __name__ == '__main__':
    main()

