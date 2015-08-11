from genomic_neuralnet.neuralnet.network import Network
from genomic_neuralnet.neuralnet.layers import LinearLayer, RbfLayer, SigmoidLayer

def get_rbf_network(num_inputs, num_hidden, num_outputs, centers=None, spread=1.):
    if num_hidden is None:
        num_hidden = 0 # Could be overwritten later.
    hidden_layer = RbfLayer(num_inputs, num_hidden, centers=centers, spread=spread)
    output_layer = LinearLayer(num_hidden, num_outputs)
    network = Network(hidden_layer, output_layer)
    return network

def get_mlp_network(num_inputs, num_hidden, num_outputs):
    hidden_layer = RbfLayer(num_inputs, num_hidden)
    output_layer = SigmoidLayer(num_hidden, num_outputs)
    network = Network(hidden_layer, output_layer)
    return network
