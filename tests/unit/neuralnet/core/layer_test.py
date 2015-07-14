import numpy as np
from genomic_neuralnet.neuralnet.core import BackpropLayer

def test_layer_weights_have_correct_shape():  
    input_nodes = 4 
    layer_nodes = 2
    layer = BackpropLayer(input_nodes, layer_nodes)

    assert layer.weights.shape == (layer_nodes, input_nodes)

def test_layer_biases_have_correct_shape():
    input_nodes = 4
    layer_nodes = 2
    layer = BackpropLayer(input_nodes, layer_nodes)

    assert layer.biases.shape == (layer_nodes,)

