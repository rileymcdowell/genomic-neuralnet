import numpy as np
from scipy.special import expit as sigmoid
from genomic_neuralnet.neuralnet.core import BackpropLayer

def sigmoid_derivative(x):
    return sigmoid(x) * (1-sigmoid(x))

class SigmoidLayer(BackpropLayer):
    def __init__(self, num_inputs, num_neurons, apply_bias=True):
        super(SigmoidLayer, self).__init__(num_inputs, num_neurons)
        self._apply_bias = apply_bias 

    def activate(self, inputs):
        """ 
        inputs = numpy array of input values, one per input neuron.
        outputs = numpy array of outputs, one per neuron in this layer.
        Outputs calculated using a sigmodal activation function.
        """
        internal_activations = np.dot(self.weights, inputs)
        if self._apply_bias:
            internal_activations += self.biases
        return sigmoid(internal_activations)

    def activate_many(self, inputs):
        """ 
        inputs = numpy array of input values, one per input neuron.
        outputs = numpy array of outputs, one per neuron in this layer.
        Outputs calculated using a sigmodal activation function.
        """
        dot_prods = np.einsum('ij,kj->ik', inputs, self.weights)
        if self._apply_bias:
            dot_prods += self.biases
        reshaped = dot_prods.reshape(inputs.shape[0], self.weights.shape[0])
        return sigmoid(reshaped)

