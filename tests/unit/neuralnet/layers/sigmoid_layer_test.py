import numpy as np
from genomic_neuralnet.neuralnet.layers import SigmoidLayer 

def test_sigmoid_layer_activates_correctly():
    input_nodes = 3 
    layer_nodes = 2
    
    inputs = np.array([0.5, 1.0, -1.5])

    layer = SigmoidLayer(input_nodes, layer_nodes)
    # Force weights + biases to static values for testing purposes.
    layer.weights = np.array([1.,2.,3.,4.,5.,6.]).reshape((layer_nodes, input_nodes))
    layer.biases = np.array([1.,2.])

    # How this _should_ work:
    # input 0.5     layer (1, 2, 3) + 1 = (0.5 + 2 - 4.5) + 1 = -1  
    #
    # input 1.0     layer (4, 5, 6) + 2 = (2 + 5 - 9)     + 2 = 0 
    #
    # input -1.5

    # Output = sigmoid((-1, 0)) == (0.2689414, 0.5)
    outputs = tuple(layer.activate(inputs))

    assert np.allclose(outputs, (0.2689414, 0.5), atol=1e-4)



