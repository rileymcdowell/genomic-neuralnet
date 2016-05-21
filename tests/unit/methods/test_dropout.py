from __future__ import print_function

import numpy as np

from genomic_neuralnet.methods import get_do_net_prediction
from genomic_neuralnet.util import NeuralnetConfig

def test_dropout_network_can_approximate_sin_function():
    x = np.mgrid[-np.pi:np.pi:1000j][:, np.newaxis]
    x_dense = np.mgrid[-np.pi:np.pi:5000j][:, np.newaxis]

    def y_func(x):
        return (np.sin(x) + 1) / 2 # Scale to [0,1].

    y = y_func(x)
    y_dense = y_func(x_dense)

    config = NeuralnetConfig()
    config.hidden_layers = (100,)
    config.learning_rate = 0.05
    config.batch_splits = 5
    config.max_epochs = 10000
    config.try_convergence = True
    config.continue_epochs = 2000

    predicted = get_do_net_prediction(x, y, x_dense, y_dense, config, dropout_keep_prob=0.95)

    # No problem getting within 0.1 of actual function. 
    all_close = np.allclose(y_dense.ravel(), predicted, rtol=0.0, atol=0.1) 
    
    if not all_close:
        try:
            import matplotlib.pyplot as plt
            plt.plot(x_dense, y_dense, color='blue')
            plt.plot(x_dense, predicted, color='red')
            plt.show()
        except:
            pass

    assert all_close

