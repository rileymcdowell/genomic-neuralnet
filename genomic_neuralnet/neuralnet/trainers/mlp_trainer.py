import numpy as np
from numpy.linalg import pinv
from genomic_neuralnet.neuralnet import SigmoidLayer, LinearLayer

class MlpTrainer(object):
    def __init__(self, mlp_network):
        # Check that this is compatible with pseudoinverse training.
        assert len(rbf_network.layers) == 2
        assert type(rbf_network.layers[0]) == SigmoidLayer
        assert type(rbf_network.layers[1]) == LinearLayer 
        self._target_network = rbf_network 
        self._sigmoid_layer = self._target_network.layers[0]
        self._output_layer = self._target_network.layers[1]

    def train_on_data(self, sample_data, sample_output):
        """
        This method will train an RBF network assuming that centers
        have already been set. The resulting weights in the output layer are set
        to minimize the ordinary least squares output error using the
        pseudoinverse weight-setting method.
        """
        weights = self._sigmoid_layer.weights
        biases = self._sigmoid_layer.biases
        for sample_idx in output_layer.num_neurons:
            
        output = sample_output # Prime the output.
        # Iterate through layers backwards (backpropagate).
        for layer in self._target_network.layers: 
            weight_gradient, bias_gradient = layer.activate_with_pre_activate(inputs)




