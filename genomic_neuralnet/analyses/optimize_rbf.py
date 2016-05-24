from __future__ import print_function
import sys
import pandas as pd
import numpy as np

from functools import partial

from genomic_neuralnet.config import JOBLIB_BACKEND
from genomic_neuralnet.common import run_predictors
from genomic_neuralnet.methods import get_rbf_nn_prediction 

# These ranges likely contain the global maximum. Find it. 
hidden_size = [50, 100, 150, 200, 250]
spread = np.arange(50, 1001, 50) 

params = [(h, s) for h in hidden_size for s in spread]
prediction_functions = map(lambda (h, s): partial(get_rbf_nn_prediction, centers=h, spread=s), params)

def main():
    df = pd.DataFrame.from_records(params, columns=['neurons', 'spread'])

    accuracies = run_predictors(prediction_functions, backend=JOBLIB_BACKEND, cycles=5)

    means = []
    std_devs = []
    for (h, s), accuracy_arr in zip(params, accuracies):
        mean_acc = np.mean(accuracy_arr)
        std_acc = np.std(accuracy_arr)
        print('{} accuracy: mean {} sd {}'.format((h,s), np.mean(accuracy_arr), np.std(accuracy_arr)))
        means.append(mean_acc)
        std_devs.append(std_acc)

    df['mean'] = means
    df['std_dev'] = std_devs
    df.to_csv('plots/optimal_rbf.csv', index=False)

    print('Done')

if __name__ == '__main__':
    main()

