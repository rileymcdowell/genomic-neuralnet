from genomic_neuralnet.neuralnet.layers import LinearLayer, RbfLayer, SigmoidLayer
from functools import reduce

class Network(object):
    def __init__(self, hidden_layer, output_layer):
        self.layers = [hidden_layer, output_layer]

    def activate(self, inputs):
        # Fold over the layers, activating them as a series.
        return reduce(lambda acc, layer: layer.activate(acc), self.layers, inputs)

    def activate_many(self, inputs):
        # Fold over the layers, activating them as a series.
        return reduce(lambda acc, layer: layer.activate_many(acc), self.layers, inputs)

