from __future__ import print_function

import sys
import pandas as pd
import numpy as np

from functools import partial

from genomic_neuralnet.util import NeuralnetConfig
from genomic_neuralnet.common import run_predictors 
from genomic_neuralnet.methods import get_do_net_prediction 

CYCLES = 1

def get_config(hidden_layers):
    config = NeuralnetConfig()
    config.learning_rate = 0.001
    config.continue_epochs = 1000
    config.hidden_layers = (hidden_layers,)
    return config

# These ranges likely contain the global maximum. Find it. 
#hidden_size = [25, 50, 100, 150, 200, 250]
#dropout_keep_probs = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
hidden_size = [20]
dropout_keep_probs = [0.75]

params = [(get_config(h), d) for h in hidden_size for d in dropout_keep_probs]

def get_partial(c, d):
    func = partial(get_do_net_prediction, net_config=c, dropout_keep_prob=d)
    return func

prediction_functions = map(lambda (c, d): get_partial(c, d), params)

def main():
    df = pd.DataFrame.from_records(params, columns=['hidden', 'dropout_keep'])

    accuracies = run_predictors(prediction_functions, cycles=CYCLES)

    means = []
    std_devs = []
    for (h, s), accuracy_arr in zip(params, accuracies):
        h = h.hidden_layers[0]
        mean_acc = np.mean(accuracy_arr)
        std_acc = np.std(accuracy_arr)
        print('{} accuracy: mean {} sd {}'.format((h,s), np.mean(accuracy_arr), np.std(accuracy_arr)))
        means.append(mean_acc)
        std_devs.append(std_acc)

    df['mean'] = means
    df['std_dev'] = std_devs
    df.to_csv('plots/optimal_do.csv', index=False)

    print('Done')

if __name__ == '__main__':
    main()

