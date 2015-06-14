import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcess

# Data collected from a pybrain/neuralnet run on the Bay-0xShahdara dataset FLOLD.
# Used a 20 cycle run with train size 90% and 0% required markers.
# 200 epochs.

# decay parameter, mean of error, stddev of error
DECAY_IDX = 0
ACCURACY_IDX = 1
STD_DEV_IDX = 2

arr = np.array([ 
      (0.005, 0.845783256895, 0.0419324147843)
    , (0.01, 0.848194306499, 0.0411759612146)
    , (0.015, 0.832825013298, 0.0445893131624)
    , (0.02, 0.821842815211, 0.0434246480306)
    , (0.025, 0.814743100871, 0.0446891849535)
    , (0.03, 0.803703321072, 0.0474043756227)
    , (0.035, 0.797347462263, 0.0545588421622)
    , (0.04, 0.799289123822, 0.0508327346284)
    , (0.045, 0.79495506474, 0.0491416502302)
    , (0.05, 0.798770226778, 0.0497335578309)

    ])


n = len(arr)

# When using a squared exponential correlation function (the default)
# the nugget values are equal to the variance of the points. That's
# just the square of the standard deviations. 
error = np.power(arr[:,2], STD_DEV_IDX) 
gp = GaussianProcess(nugget=error)
input = arr[:, DECAY_IDX].reshape((n,1))
output = arr[:, ACCURACY_IDX]

# Fit gaussian process model.
gp.fit(input, output)

# Predict the results for intermediate values from 1 to 32.
lower = np.min(arr[:, DECAY_IDX])
upper = np.max(arr[:, DECAY_IDX])
x = np.arange(lower, upper, 0.001)
x = x.reshape((x.shape[0], 1))
y_pred, mse = gp.predict(x, eval_MSE=True)
sigma = np.sqrt(mse)

# Plot the results.
fig = plt.figure()

plt.plot(x, y_pred)
plt.errorbar(input.ravel(), output, error, fmt='r.', markersize=10)
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=0.5, fc='b', ec='None', label='95% confidence interval')

title_1 = 'Gaussian process regression estimation of optimal'
title_2 = 'decay parameter for two neuron hidden layer'
plt.title('\n'.join([title_1, title_2]))
plt.xlabel('Weight decay parameter')
plt.ylabel('Correlation with measured phenotype')

plt.show()
plt.savefig('optimal_hidden_layer.png')
