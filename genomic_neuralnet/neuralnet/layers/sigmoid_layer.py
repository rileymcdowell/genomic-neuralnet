import numpy as np
from scipy.special import expit as sigmoid
from genomic_neuralnet.neuralnet.core import BackpropLayer

def sigmoid_derivative(x):
    return sigmoid(x) * (1-sigmoid(x))

class SigmoidLayer(BackpropLayer):
    def __init__(self, num_inputs, num_neurons):
        super(SigmoidLayer, self).__init__(num_inputs, num_neurons)

    def activate(self, inputs):
        """ 
        inputs = numpy array of input values, one per input neuron.
        outputs = numpy array of outputs, one per neuron in this layer.
        Outputs calculated using a sigmodal activation function.
        """
        return sigmoid(np.dot(self.weights, inputs) + self.biases)

    def activate_many(self, inputs):
        """ 
        inputs = numpy array of input values, one per input neuron.
        outputs = numpy array of outputs, one per neuron in this layer.
        Outputs calculated using a sigmodal activation function.
        """
        dot_prods = np.einsum('ij,kj->ik', inputs, self.weights)
        with_biases = dot_prods + self.biases
        reshaped = with_biases.reshape(inputs.shape[0], self.weights.shape[0])
        return sigmoid(reshaped)

    #def get_weight_bias_gradient(self, inputs, outputs):
    #    error = self.activate(inputs) - outputs 
