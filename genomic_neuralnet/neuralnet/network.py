from genomic_neuralnet.neuralnet.layers import LinearLayer, RbfLayer, SigmoidLayer
from functools import reduce

class Network(object):
    def __init__(self, hidden_layer, output_layer):
        self.layers = [hidden_layer, output_layer]

    def activate(self, input):
        # Fold over the layers, activating them as a series.
        return reduce(lambda acc, layer: layer.activate(acc), self.layers, input)

    def activate_many(self, inputs):
        # Fold over the layers, activating them as a series.
        return reduce(lambda acc, layer: layer.activate_many(acc), self.layers, inputs)

    def activate_include_layer_inputs(self, input):
        """
        Like activate, but returns all of the intermediate inputs and outputs for
        the network instead of just the final value.
        """
        # Iterate over the layers, activating them in series and collecting all state.
        layer_io = []
        input = input
        for layer in self.layers:
            output = layer.activate(input)
            layer_io.append((input, output))
            input = output # Pass output of last layer into input of next layer.
        return layer_io

