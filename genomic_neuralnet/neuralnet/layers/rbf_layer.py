import numpy as np
import numpy.linalg as la
from scipy.special import expit 
from genomic_neuralnet.neuralnet.core import Layer
from genomic_neuralnet.util import require_true 

_REQUIRE_TRUE_MESSAGE = 'Must have centers to activate RBF layer'

class RbfLayer(Layer):
    def __init__(self, num_inputs, num_neurons, centers=None, spread=1.):
        super(RbfLayer, self).__init__(num_inputs, num_neurons)
        if not centers is None:
            assert centers.shape == (num_neurons, num_inputs) or centers.shape == (num_neurons,)
        self.centers = centers 
        self.spread = spread
        self.beta = 1.

    @require_true(lambda self: not self.centers is None, _REQUIRE_TRUE_MESSAGE)
    def activate(self, inputs):
        """ 
        inputs = numpy array of input values, one per input neuron.
        outputs = numpy array of outputs, one per neuron in this layer.
        Outputs calculated using an RBF.
        """
        norm = la.norm(inputs - self.centers, axis=1)
        out = np.exp(-self.beta * norm**2 / (2*self.spread))
        return out

    @require_true(lambda self: not self.centers is None, _REQUIRE_TRUE_MESSAGE)
    def activate_many(self, inputs):
        """ 
        inputs = numpy array of input values, one per input neuron.
        outputs = numpy array of outputs, one per neuron in this layer.
        Outputs calculated using an RBF.
        """
        # Tile and repeat for easy calculation of euclidean distance.
        inputs_repeat = np.repeat(inputs, self.centers.shape[0], axis=0)
        centers_tile = np.tile(self.centers.T, inputs.shape[0])
        pairs = inputs_repeat.T - centers_tile # Easy to do on tiled data.
        norm = np.sqrt(np.sum(pairs**2, axis=0)).reshape(inputs.shape[0], self.centers.shape[0])
        out = np.exp(-self.beta * norm**2 / (2*self.spread))
        return out
