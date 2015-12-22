from __future__ import print_function
import copy
import numpy as np
from numpy.linalg import pinv
from genomic_neuralnet.neuralnet import SigmoidLayer, LinearLayer

class MlpTrainer(object):
    def __init__(self, mlp_network, learning_rate=1.0):
        # Check that this is compatible with pseudoinverse training.
        assert len(mlp_network.layers) == 2
        assert type(mlp_network.layers[0]) == SigmoidLayer
        assert type(mlp_network.layers[1]) in (LinearLayer, SigmoidLayer)
        self._target_network = mlp_network 
        self._hidden_layer = self._target_network.layers[0]
        self._output_layer = self._target_network.layers[1]
        self._learning_rate = learning_rate

    def train_on_data(self, sample_data, sample_output):
        """
        This method will train an MLP network using a standard
        gradient descent techinque on a list of inputs and their 
        associated, ground truth outputs. 
        """
        if self._hidden_layer._apply_bias:
            raise NotImplementedError('Code not written for training to handle bias yet')

        layer_weights = map(lambda x: x.weights, self._target_network.layers)

        sample = sample_data
        desired_out = sample_output

        layer_io = self._target_network.activate_many_include_layer_inputs(sample)
        # Find output slope times the difference between network activation and truth. 
        # sigmoid_prime(x) = sigmoid(x) * (1 - sigmoid(x))
        # difference = desired_out - last_output
        # output error = (t-o)*o*(1-o)
        # Most people refer to this whole quanitity as the 'output error'.
        last_output = layer_io[-1][1] # Last layer's output.
        last_error = (desired_out - last_output) * last_output * (1 - last_output)
        for layer_idx, (layer_input, layer_output) in reversed(list(enumerate(layer_io))):
            # Adjust weights based on error. Average them across all samples.
            #print('la', last_error.shape)
            #print('li', layer_input.shape)
            products = last_error * layer_input
            #print('la x li', products.shape)
            weight_adjustments = np.mean(last_error * layer_input, axis=0)
            #print('wa', weight_adjustments.shape)
            #print('lw', layer_weights[layer_idx].shape)
            layer_weights[layer_idx] += weight_adjustments 
            # Use the new weights to adjust preceeding layer's outputs.
            if not layer_idx == 0:
                last_error = last_error * layer_weights[layer_idx] * layer_output * (1 - layer_output)
            #print()
        
    def train_on_datum(self, sample_datum, sample_output):
        """
        This method will train an MLP network using a standard
        gradient descent techinque on a single (input, output) pair. 
        """
        if self._hidden_layer._apply_bias:
            raise NotImplementedError('Code not written for training to handle bias yet')

        layer_weights = map(lambda x: x.weights, self._target_network.layers)

        sample = sample_datum
        desired_out = sample_output
        
        # Layer IO = [(in, out), ..., (in, out)]
        layer_io = self._target_network.activate_include_layer_inputs(sample)

        last_output = layer_io[-1][1] # Last layer's output.
        # TODO: Determine linear layer derivative.

        # Calulate output slope times the difference between network activation and truth.
        #
        # Forumlas: 
        # sigmoid_prime(x) = sigmoid(x) * (1 - sigmoid(x))
        # difference = desired_out - last_output
        # "output error" = difference * sigmoid_prime(x)
        # "output error" = (t-o) * o(1-o)
        # 
        # Most people refer to this whole quanitity as the 'output error', although 
        # it's actually # error * gradient.
        error = (desired_out - last_output) * last_output * (1 - last_output)

        # Backpropagate over the layers.
        for layer_idx, (layer_input, layer_output) in reversed(list(enumerate(layer_io))):
            layer_shape = layer_weights[layer_idx].shape
            # Adjust weights based on error.
            repeated_error = np.repeat(error, layer_shape[1]).reshape(layer_shape)
            tiled_inputs = np.tile(layer_input, layer_shape[0]).reshape(layer_shape)
            layer_weights[layer_idx] += repeated_error * tiled_inputs * self._learning_rate 
            # Calculate the previous layer's error, but only if there is a previous layer. 
            if not layer_idx == 0:
                # Repeat for each hidden layer neuron.
                reshaped_error = np.repeat(error, layer_shape[1]).reshape(layer_shape)
                error = reshaped_error * layer_weights[layer_idx] * layer_input * (1 - layer_input)
                # Sum over output neurons.
                error = np.sum(error.T, axis=1)

