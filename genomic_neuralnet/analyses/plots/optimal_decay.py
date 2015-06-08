import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcess

# Data collected from a pybrain/neuralnet run on the Bay-0xShahdara dataset FLOLD.
# Used a 50 cycle run with train size 90% and 0% required markers.
# 500 max epocs, 50 continue epochs.

# decay parameter, mean of error, stddev of error
arr = np.array([ (0.01, 0.657945630935, 0.0761733137313)
               , (0.05, 0.689541149762, 0.0733185094002)
               , (0.1, 0.678116229027, 0.0754871574157)
               , (0.2, 0.669201514574, 0.100304186167)
               , (0.4, 0.65343506879, 0.101377871979)
               , (0.5, 0.656056348646, 0.091372006148)
               ])


n = len(arr)
sample_n = 500

# Divide by sqrt(sample_n) to get standard error of the mean. 
error = arr[:,2]/np.sqrt(sample_n) 
gp = GaussianProcess(nugget=error)
input = arr[:,0].reshape((n,1))
output = arr[:,1]

# Fit gaussian process model.
gp.fit(input, output)

# Predict the results for intermediate values from 1 to 32.
x = np.arange(0, 0.5, 0.01)
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
plt.xlabel('Number of hidden layer neurons')
plt.ylabel('Correlation with measured phenotype')

plt.show()
plt.savefig('optimal_hidden_layer.png')
