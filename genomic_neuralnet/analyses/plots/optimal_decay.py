import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcess

# Data collected from a pybrain/neuralnet run on the Bay-0xShahdara dataset FLOLD.
# Used a 50 cycle run with train size 90% and 0% required markers.
# 500 max epocs, 50 continue epochs.

# decay parameter, mean of error, stddev of error
arr = np.array([ (0.01, 0.781274022734, 0.0410609485148)
               , (0.015, 0.78253626307, 0.0386762378667)
               , (0.02, 0.789138908225, 0.0379565273988)
               , (0.025, 0.788893278602, 0.0333444387478)
               , (0.03, 0.784581930656, 0.0385815200278)
               ])


n = len(arr)
sample_n = 25 

# Divide by sqrt(sample_n) to get standard error of the mean. 
error = arr[:,2]/np.sqrt(sample_n) 
gp = GaussianProcess(nugget=error)
input = arr[:,0].reshape((n,1))
output = arr[:,1]

# Fit gaussian process model.
gp.fit(input, output)

# Predict the results for intermediate values from 1 to 32.
x = np.arange(0.00, 0.04, 0.001)
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
title_2 = 'decay parameter for single hidden layer'
plt.title('\n'.join([title_1, title_2]))
plt.xlabel('Weight decay parameter')
plt.ylabel('Correlation with measured phenotype')

plt.show()
plt.savefig('optimal_hidden_layer.png')
