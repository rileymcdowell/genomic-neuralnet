import numpy as np
from genomic_neuralnet.neuralnet.core import Layer

class BackpropLayer(Layer):
    def __init__(self, num_inputs, num_neurons):
        super(BackpropLayer, self).__init__(num_inputs, num_neurons)
        # Randomly initialize weights.
        self.weights = np.random.randn(self._num_inputs, self._num_neurons)
        self.weights = self.weights.reshape((self._num_neurons, self._num_inputs))
        self.biases = np.random.randn(self._num_neurons)

