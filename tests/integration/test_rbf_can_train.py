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

    rbf_network = get_rbf_network(1, num_centers, 1, centers=centers, spread=1.)
    rbf_trainer = RbfTrainer(rbf_network)

    print(rbf_network.layers[0].centers)
    rbf_trainer.train_on_data(x, y)

    predicted_out = [] 
    predicted_out = rbf_network.activate_many(x_dense)

    # No problem getting within 0.1 of actual function. 
    assert np.allclose(y_dense, predicted_out, rtol=0.0, atol=1e-1) 
     
def test_can_select_centers_for_multiple_inputs_ordinary_least_squares():
    """
    Specifically test for training with the non-fast ordinary least squares 
    training method.
    """
    inputs = 2
    outputs = 1
    x, y = np.mgrid[-1:1:15j, -1:1:15j]
    x_dense, y_dense = np.mgrid[-1:1:30j, -1:1:30j]
    def z_func(x, y):
        return np.sin(x) / 2  + np.cos(y)
    z = z_func(x, y)
    z_dense = z_func(x_dense, y_dense)

    rbf_network = get_rbf_network(inputs, 1, outputs, centers=None, spread=8.)
    rbf_trainer = RbfTrainer(rbf_network)

    xy = np.column_stack([x.flatten(), y.flatten()])
    z = z.reshape((np.prod(z.shape), 1))
    rbf_trainer.train_with_best_centers(xy, z, num_centers=4, fast_ols=False)

    xy_dense = np.column_stack([x_dense.flatten(), y_dense.flatten()])
    predicted_out = rbf_network.activate_many(xy_dense)

    # No problem getting within 0.1 of actual function. 
    print(np.max(np.abs(z_dense.flatten() - predicted_out.flatten())))
    assert np.allclose(z_dense.flatten(), predicted_out.flatten(), rtol=0.0, atol=1e-1) 

def test_can_train_with_ols_algorithm():
    inputs = 2
    outputs = 1
    x, y = np.mgrid[-1:1:15j, -1:1:15j]
    x_dense, y_dense = np.mgrid[-1:1:30j, -1:1:30j]
    def z_func(x, y):
        return np.sin(x) / 2  + np.cos(y)
    z = z_func(x, y)
    z_dense = z_func(x_dense, y_dense)

    rbf_network = get_rbf_network(inputs, 1, outputs, centers=None, spread=8.)
    rbf_trainer = RbfTrainer(rbf_network)

    xy = np.column_stack([x.flatten(), y.flatten()])
    z = z.reshape((np.prod(z.shape), 1))
    rbf_trainer.train_with_best_centers(xy, z, num_centers=8)

    xy_dense = np.column_stack([x_dense.flatten(), y_dense.flatten()])
    predicted_out = rbf_network.activate_many(xy_dense)

    # No problem getting within 0.1 of actual function. 
    print(np.max(np.abs(z_dense.flatten() - predicted_out.flatten())))
    assert np.allclose(z_dense.flatten(), predicted_out.flatten(), rtol=0.0, atol=1e-1) 
