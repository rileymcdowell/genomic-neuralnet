from __future__ import print_function
import sys
import pandas as pd
import numpy as np

from functools import partial

from genomic_neuralnet.common import run_predictors 
from genomic_neuralnet.methods import get_nn_dom_prediction 

# These ranges likely contain the global maximum. Find it. 
decay_step = 0.002
decay_size = list(np.arange(0.00, 0.0501, decay_step))

params = [(h, wd) for h in layer_size for wd in decay_size]
prediction_functions = map(lambda (h, wd): partial(get_nn_dom_prediction, hidden=(h,), weight_decay=wd), params)

def main():
    df = pd.DataFrame.from_records(params, columns=['neurons', 'weight_decay'])

    accuracies = run_predictors(prediction_functions)

    means = []
    std_devs = []
    for (h, wd), accuracy_arr in zip(params, accuracies):
        mean_acc = np.mean(accuracy_arr)
        std_acc = np.std(accuracy_arr)
        print('{} accuracy: mean {} sd {}'.format((h,wd), np.mean(accuracy_arr), np.std(accuracy_arr)))
        means.append(mean_acc)
        std_devs.append(std_acc)

    df['mean'] = means
    df['std_dev'] = std_devs
    df.to_csv('plots/optimal_dominance.csv', index=False)

    print('Done')

if __name__ == '__main__':
    main()

