import numpy as np
from genomic_neuralnet.neuralnet.core import BackpropLayer

class LinearLayer(BackpropLayer):
    def __init__(self, num_inputs, num_neurons):
        super(LinearLayer, self).__init__(num_inputs, num_neurons)

    def activate(self, inputs):
        """ 
        inputs = numpy array of input values, one per input neuron.
        outputs = numpy array of outputs, one per neuron in this layer.
        Outputs calculated without any activation function applied.
        """
        return np.dot(self.weights, inputs) + self.biases

    def activate_many(self, inputs):
        """ 
        inputs = numpy array of input values, one per input neuron.
        outputs = numpy array of outputs, one per neuron in this layer.
        Outputs calculated without any activation function applied.
        """
        dot_prods = np.einsum('ij,kj->ik', inputs, self.weights)
        with_biases = dot_prods + self.biases
        return with_biases.reshape(inputs.shape[0], self.weights.shape[0])
