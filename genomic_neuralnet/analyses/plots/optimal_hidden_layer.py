import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcess

# Data collected from a pybrain/neuralnet run on the Bay-0xShahdara dataset FLOLD.
# Think I used a 32 cycle run with train size 90% and 80% required markers.

# hidden layer neurons, mean of error, stddev of error
arr = np.array( [ (1,  0.661107314277, 0.118788414527)
                , (2,  0.676115939485, 0.122306463591)
                , (4,  0.610622067516, 0.118818108442)
                , (8,  0.599599535789, 0.0917628474102)
                , (16, 0.531216661002, 0.11498933628)
                , (32, 0.519561320092, 0.0954161187351)
                ])

n = len(arr)

# Divide by sqrt(n) to get standard error of the mean. 
error = arr[:,2]/np.sqrt(n) 
gp = GaussianProcess(nugget=error)
input = arr[:,0].reshape((n,1))
output = arr[:,1]

# Fit gaussian process model.
gp.fit(input, output)

# Predict the results for intermediate values from 1 to 32.
x = np.arange(1, 32, 0.1)
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

title_1 = 'Gaussian process regression estimation of optimal hidden'
title_2 = 'layer neuron count for single hidden layer'
plt.title('\n'.join([title_1, title_2]))
plt.xlabel('Number of hidden layer neurons')
plt.ylabel('Correlation with measured phenotype')

plt.show()
plt.savefig('optimal_hidden_layer.png')
