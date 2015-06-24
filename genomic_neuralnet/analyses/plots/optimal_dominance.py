import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process import GaussianProcess

# Data collected from a pybrain/neuralnet run on the Bay-0xShahdara dataset FLOLD.
# Used a 20 cycle run with train size 90% and 0% required markers.
# 200 epochs.

# decay parameter, mean of error, stddev of error
DECAY_IDX = 0
ACCURACY_IDX = 1
STD_DEV_IDX = 2
X_RESOLUTION = 500

this_dir = os.path.dirname(__file__)
decay_csv_path = os.path.join(this_dir, 'optimal_dominance.csv')
df = pd.DataFrame.from_csv(decay_csv_path, index_col=None)
df = df[df['weight_decay'] > 0.0] # Remove 0 values - these are outliers.
neurons = list(df['neurons'].unique())
num_neurons = len(neurons)
num_decays = len(df[df['neurons'] == 1]) 
df.set_index = ['neurons']

# When using a squared exponential correlation function (the default)
# the nugget values are equal to the variance of the points. That's
# just the square of the standard deviations. 

predictions = np.zeros((num_neurons, X_RESOLUTION))
errors = np.zeros((num_neurons, X_RESOLUTION))
sub_dfs = []
for neuron_num in neurons:
    sub_df = df[df['neurons'] == neuron_num]
    neuron_idx = neuron_num - 2
    error = np.power(sub_df['std_dev'], 2) 
    n = len(error)
    gp = GaussianProcess(nugget=error)
    input = sub_df['weight_decay'].values.reshape((n, 1))
    output = sub_df['mean'] 

    # Fit gaussian process model.
    gp.fit(input, output)

    # Predict the results for intermediate values. 
    lower = sub_df['weight_decay'].min() 
    upper = sub_df['weight_decay'].max() 
    x = np.linspace(lower, upper, num=X_RESOLUTION)
    x = x[:, np.newaxis]
    y_pred, mse = gp.predict(x, eval_MSE=True)
    sigma = np.sqrt(mse)
    
    predictions[neuron_idx,:] = y_pred
    errors[neuron_idx,:] = sigma 
    sub_dfs.append(sub_df)

# Plot the results.
sns.set_style('darkgrid')
f, axarr = plt.subplots(2, 3, sharex=True, sharey=True)

for neuron_num in neurons:
    neuron_idx = neuron_num - 2 
    ax_idx1 = neuron_idx / 3
    ax_idx2 = neuron_idx % 3 
    ax = axarr[ax_idx1, ax_idx2]
    y_pred = predictions[neuron_idx]
    sigma = errors[neuron_idx]
    output = sub_dfs[neuron_idx]['mean']
    error = sub_dfs[neuron_idx]['std_dev']

    # Remember x from the previous loop.
    #ax.plot(x, y_pred)
    ax.errorbar(input.ravel(), output, error/np.sqrt(64), fmt='r.', markersize=10)
    #ax.fill(np.concatenate([x, x[::-1]]),
    #         np.concatenate([y_pred - 1.9600 * sigma,
    #                        (y_pred + 1.9600 * sigma)[::-1]]),
    #         alpha=0.5, fc='b', ec='None')
    ax.set_title('{} nodes in hidden layer'.format(neuron_num))
    ax.set_xlabel('Weight Decay Parameter')
    ax.set_ylabel('Prediction Accuracy (correlation)')
    ax.axhline(y=0.85, color='k', linewidth=2)

title_1 = 'Multi-layer perceptron neural network predictive accuracy'
title_2 = 'when run on training set containing two entries per SNP.'
plt.suptitle('\n'.join([title_1, title_2]), y=0.02, verticalalignment='bottom')
plt.show()
plt.savefig('optimal_dominance.png')
