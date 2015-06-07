from __future__ import print_function
import sys
import numpy as np

from functools import partial

from genomic_neuralnet.common import run_predictors 
from genomic_neuralnet.methods import get_nn_prediction

prediction_functions = [ partial(get_nn_prediction, hidden=(1,))
                       , partial(get_nn_prediction, hidden=(2,))
                       , partial(get_nn_prediction, hidden=(3,))
                       , partial(get_nn_prediction, hidden=(4,))
                       ]
prediction_names = tuple(['nn {}'.format(x + 1) for x in range(len(prediction_functions))])

def main():
    accuracies = run_predictors(prediction_functions)

    print('')
    for name, accuracy_arr in zip(prediction_names, accuracies):
        print('{} accuracy: mean {} sd {}'.format(name, np.mean(accuracy_arr), np.std(accuracy_arr)))
    print('Done')

if __name__ == '__main__':
    main()

