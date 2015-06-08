import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcess

# 90% training size, 40 cycles, 200 epochs, never terminating early.
# Run on arabidopsis FLOSD dataset.

arr = np.array([ [1,0.5 , 0.651223966315,  0.0893278132106]
               , [1,1   , 0.640362768242,  0.0917147926869]
               , [1,2   , 0.661877155681,  0.0694640846404]
               , [1,4   , 0.636788963475,  0.10020489193]
               , [2,0.5 , 0.645374913279,  0.0929172925127]
               , [2,1   , 0.665254760774,  0.0985269051634]
               , [2,2   , 0.666725687582,  0.0829256637072]
               , [2,4   , 0.681823123638,  0.0854741947383]
               , [3,0.5 , 0.678657957517,  0.067249246982]
               , [3,1   , 0.653705029239,  0.0941905975165]
               , [3,2   , 0.659774488039,  0.0873911733062]
               , [3,4   , 0.667767538458,  0.102949291532]
               , [4,0.5 , 0.685795350648,  0.0853564129251]
               , [4,1   , 0.679539170347,  0.0769151933888]
               , [4,2   , 0.656663388684,  0.0896046417041]
               , [4,4   , 0.666850104132,  0.0890044045691]
               ])

# hidden layer neurons, mean of error, stddev of error
arr = arr[:,2].reshape((4,4))

# Plot the results.
fig = plt.figure()

plt.imshow(arr, interpolation='none')
title = 'Neuralnet prediction accuracy'
plt.title(title)
plt.xlabel('Hidden layer neurons')
plt.ylabel('Decay parameter')
plt.colorbar()

plt.show()
plt.savefig('optimal_layer_decay.png')
