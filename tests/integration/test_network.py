import numpy as np
from genomic_neuralnet.neuralnet import LinearLayer, SigmoidLayer, Network

def test_can_activate_network():
    input_num = 3
    hidden_num = 2
    output_num = 1
    hidden_layer = SigmoidLayer(input_num, hidden_num)
    output_layer = LinearLayer(hidden_num, output_num)

    network = Network(hidden_layer, output_layer)

    inputs = np.array([0.5, 1.0, -1.5])
    hidden_layer.weights = np.array([1.,2.,3.,4.,5.,6.]).reshape(hidden_num, input_num)
    hidden_layer.biases = np.array([1.,2.])

    # How this _should_ work:
    # input 0.5     layer (1, 2, 3) + 1 = (0.5 + 2 - 4.5) + 1 = -1
    #
    # input 1.0     layer (4, 5, 6) + 2 = (2 + 5 - 9)     + 2 = 0 
    #
    # input -1.5

    # Output = sigmoid((-1, 0)) == (0.268914, 0.5)

    output_layer.weights = np.array([2., 4.]).reshape((output_num, hidden_num))
    output_layer.biases = np.array([-2])

    # How this _should_ work:
    # Input 0.268914  layer (2, 4) - 2 = (2 * 0.26891 + 4 * 0.5) - 2 = 0.53782
    #
    # Input 0.5       

    # Output = 0.53782 # Just one output neuron
  
    output = network.activate(inputs)

    assert np.allclose([output], [0.53782], atol=1e-4) 
    


