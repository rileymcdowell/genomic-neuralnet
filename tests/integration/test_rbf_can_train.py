from __future__ import print_function
import numpy as np

from genomic_neuralnet.neuralnet import get_rbf_network
from genomic_neuralnet.neuralnet.trainers import RbfTrainer

from sklearn.cluster import KMeans

def test_can_approximate_sin_function():
    x = np.mgrid[-np.pi:np.pi:50j][:, np.newaxis]
    x_dense = np.mgrid[-np.pi:np.pi:500j][:, np.newaxis]
    def y_func(x):
        return (np.sin(x) + 1) / 2 # Scale to [0,1].
    y = y_func(x)
    y_dense = y_func(x_dense)

    num_centers = 8
    km = KMeans(n_clusters = 8)
    km.fit(x)
    centers = km.cluster_centers_.reshape((num_centers, 1))

    rbf_network = get_rbf_network(1, num_centers, 1, centers, spread=1.)
    rbf_trainer = RbfTrainer(rbf_network)

    rbf_trainer.train_on_data(x, y)

    predicted_out = [] 
    predicted_out = rbf_network.activate_many(x_dense)

    # No problem getting within 0.1 of actual function. 
    assert np.allclose(y_dense, predicted_out, rtol=0.0, atol=1e-1) 
     
