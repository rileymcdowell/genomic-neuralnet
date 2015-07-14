import numpy as np
from numpy.linalg import pinv
from genomic_neuralnet.neuralnet import RbfLayer, LinearLayer

class RbfTrainer(object):
    def __init__(self, rbf_network):
        # Check that this is compatible with pseudoinverse training.
        assert len(rbf_network.layers) == 2
        assert type(rbf_network.layers[0]) == RbfLayer 
        assert type(rbf_network.layers[1]) == LinearLayer 
        self._target_network = rbf_network 

    def train_on_data(self, sample_data, sample_output):
        rbf_layer = self._target_network.layers[0] 
        output_layer = self._target_network.layers[1]
        output_layer.biases = 0 # Don't use these right now.
        centers = rbf_layer.centers

        # First try...
        activations = rbf_layer.activate_many(sample_data)
        pseudoinverse = pinv(activations)
        weights = np.dot(pseudoinverse, sample_output)

        output_layer.weights = weights.T


