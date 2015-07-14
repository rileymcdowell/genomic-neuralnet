
class Layer(object):
    def __init__(self, num_inputs, num_neurons):
        self._num_inputs = num_inputs
        self._num_neurons = num_neurons 

    def activate(self, inputs):
        raise NotImplementedError()

    def activate_many(self, inputs):
        raise NotImplementedError()
