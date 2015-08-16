from __future__ import print_function
import numpy as np
import pandas as pd

from functools import partial
from genomic_neuralnet.common import run_predictors
from genomic_neuralnet.methods import \
        get_brr_prediction, get_en_prediction, \
        get_lasso_prediction, get_lr_prediction, \
        get_nn_prediction, get_rr_prediction, \
        get_fast_nn_prediction, get_fast_nn_dom_prediction, \
        get_nn_dom_prediction, get_rbf_nn_prediction, \
        get_sgd_prediction

optimal_nn = partial(get_nn_prediction, hidden=(2,), weight_decay=0.008)
optimal_dom = partial(get_nn_dom_prediction, hidden=(2,), weight_decay=0.008)
optimal_lasso = partial(get_lasso_prediction, alpha=0.06)
optimal_en = partial(get_en_prediction, alpha=0.10)
optimal_rbf = partial(get_rbf_nn_prediction, centers=200, spread=150)
optimal_rr = partial(get_rr_prediction, alpha=500)

prediction_functions = [ optimal_nn
                       , get_fast_nn_prediction
                       , get_fast_nn_dom_prediction
                       , optimal_rr 
                       , get_lr_prediction 
                       , get_brr_prediction
                       , optimal_lasso 
                       , optimal_en 
                       , optimal_rbf
                       , get_sgd_prediction
                       ]

prediction_names     = [ 'wdecay_mlp_nn'
                       , 'std_mlp_nn'
                       , 'dom_mlp_nn'
                       , 'ridge_reg'
                       , 'linear_reg'
                       , 'baesian_ridge_reg'
                       , 'lasso_reg'
                       , 'elastic_net'
                       , 'rbf_nn'
                       , 'sgd_reg'
                       ]

def main():
    df = pd.DataFrame.from_records(map(lambda x: tuple([x]), prediction_names), columns=['method'])

    accuracies = run_predictors(prediction_functions)

    means = []
    std_devs = []
    for name, accuracy_arr in zip(prediction_names, accuracies):
        mean_acc = np.mean(accuracy_arr)
        std_acc = np.std(accuracy_arr)
        print('{} accuracy: mean {} sd {}'.format(name, mean_acc, std_acc))
        means.append(mean_acc)
        std_devs.append(std_acc)

    df['mean'] = means
    df['std_dev'] = std_devs
    df.to_csv('plots/compare_predictors.csv', index=False)

    print('Done')

if __name__ == '__main__':
    main()

