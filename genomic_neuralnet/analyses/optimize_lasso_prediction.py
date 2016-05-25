from __future__ import print_function
import numpy as np

from functools import partial
from genomic_neuralnet.config import JOBLIB_BACKEND
from genomic_neuralnet.common import run_predictors
from genomic_neuralnet.methods import \
        get_brr_prediction, get_en_prediction, \
        get_lasso_prediction, get_lr_prediction, \
        get_nn_prediction, get_rr_prediction

prediction_functions = []
alphas = list(np.arange(0.01, 1.01,0.01))

for x in alphas:
    prediction_functions.append(partial(get_lasso_prediction, alpha=x))

prediction_names = ['lasso_alpha={}'.format(x) for x in alphas]

def main():
    accuracies = run_predictors(prediction_functions, backend=JOBLIB_BACKEND)

    print('')
    for name, accuracy_arr in zip(prediction_names, accuracies):
        print('{} accuracy: mean {} sd {}'.format(name, np.mean(accuracy_arr), np.std(accuracy_arr)))
    print('Done')

if __name__ == '__main__':
    main()

