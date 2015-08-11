import numpy as np
from scipy.special import expit 
from genomic_neuralnet.neuralnet.core import BackpropLayer

def sigmoid_derivative(x):
    return expit(x) * (1-expit(x))

class SigmoidLayer(BackpropLayer):
    def __init__(self, num_inputs, num_neurons):
        super(SigmoidLayer, self).__init__(num_inputs, num_neurons)

    def activate(self, inputs):
        """ 
        inputs = numpy array of input values, one per input neuron.
        outputs = numpy array of outputs, one per neuron in this layer.
        Outputs calculated using a sigmodal activation function.
        """
        return expit(np.dot(self.weights, inputs) + self.biases)

    def get_weight_bias_gradient(self, inputs, outputs):
        error = self.activate(inputs) - outputs 
