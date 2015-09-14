import copy
import numpy as np
from numpy.linalg import pinv
from genomic_neuralnet.neuralnet import SigmoidLayer, LinearLayer

class MlpTrainer(object):
    def __init__(self, mlp_network):
        # Check that this is compatible with pseudoinverse training.
        assert len(mlp_network.layers) == 2
        assert type(mlp_network.layers[0]) == SigmoidLayer
        assert type(mlp_network.layers[1]) in (LinearLayer, SigmoidLayer)
        self._target_network = mlp_network 
        self._hidden_layer = self._target_network.layers[0]
        self._output_layer = self._target_network.layers[1]

    def train_on_datum(self, sample_data, sample_output):
        """
        This method will train an MLP network using a standard
        gradient descent techinque. 
        """
        if self._hidden_layer._apply_bias:
            raise NotImplementedError('Code not written for training to handle bias yet')

        layer_weights = map(lambda x: x.weights, self._target_network.layers)
        new_weights = copy.deepcopy(layer_weights) # Initialize to old value.

        sample = sample_data
        desired_out = sample_output
        
        layer_io = self._target_network.activate_include_layer_inputs(sample)
        # Output slope times the difference between network activation and truth. 
        # sigmoid_prime(x) = sigmoid(x) * (1 - sigmoid(x))
        # Most people refer to this whole quanitity as the 'output error'.
        last_output = layer_io[-1][1] # Last layer's output.
        last_error = (desired_out - last_output) * last_output * (1 - last_output)
        for layer_idx, (layer_input, layer_output) in reversed(list(enumerate(layer_io))):
            # Adjust weights based on error.
            new_weights[layer_idx] += (last_error * layer_input)
            # Use the new weights to adjust preceeding layer's outputs.
            last_error = last_error * new_weights[layer_idx] * layer_output * (1 - layer_output)

        for layer_idx, weights in enumerate(new_weights):
            self._target_network.layers[layer_idx].weights = weights
        

