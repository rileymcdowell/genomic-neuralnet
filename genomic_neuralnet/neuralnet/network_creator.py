from genomic_neuralnet.neuralnet.network import Network
from genomic_neuralnet.neuralnet.layers import LinearLayer, RbfLayer, SigmoidLayer

def get_rbf_network(num_inputs, num_hidden, num_outputs, centers=None, spread=1.):
    if num_hidden is None:
        num_hidden = 0 # Could be overwritten later.
    hidden_layer = RbfLayer(num_inputs, num_hidden, centers=centers, spread=spread)
    output_layer = LinearLayer(num_hidden, num_outputs)
    network = Network(hidden_layer, output_layer)
    return network

def get_mlp_network_regressor(num_inputs, num_hidden, num_outputs, apply_bias=True):
    """
    The mlp network regressor has linear neurons in the output
    layer, and thus has an unrestricted output range, which is
    well suited for regression.
    """
    hidden_layer = SigmoidLayer(num_inputs, num_hidden, apply_bias)
    output_layer = LinearLayer(num_hidden, num_outputs)
    network = Network(hidden_layer, output_layer)
    return network

def get_mlp_network_classifier(num_inputs, num_hidden, num_outputs, apply_bias=True):
    """
    The mlp network classifier has sigmoid neurons in the output
    layer, and thus has a limited range (0 < x < 1). This is
    well suited for classification problems where boolean
    output or outputs are desired.
    """
    hidden_layer = SigmoidLayer(num_inputs, num_hidden, apply_bias)
    output_layer = SigmoidLayer(num_hidden, num_outputs, apply_bias)
    network = Network(hidden_layer, output_layer)
    return network
