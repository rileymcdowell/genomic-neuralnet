import numpy as np
from genomic_neuralnet.neuralnet.layers import RbfLayer

def test_can_activate_on_multiple_samples_with_complex_shape():
    # Dead simple inputs. The key is that output arrays are correctly shaped.
    input_nodes = 2 
    layer_nodes = 3
    
    centers = np.array([5., 4., 3., 2., 1., 0.])
    centers = centers.reshape((layer_nodes, input_nodes))

    layer = RbfLayer(input_nodes, layer_nodes, centers, spread=1.5)
    layer.beta = 1.

    inputs = np.array([[-1., 0.], [0., 1.], [1., 2.], [2., 3.], [-2, -1]])

    # 2 input values, 3 hidden neurons, 5 samples. Shouldn't get much for broadcasting
    outputs = layer.activate_many(inputs)

    # Should produce 3 values for each of the 5 inputs.
    assert outputs.shape == (5, 3)  

def test_can_activate_many():
    # Dead simple inputs. The key is that output arrays are correctly shaped.
    input_nodes = 1
    layer_nodes = 1
    
    centers = np.array([0])
    centers = centers.reshape((layer_nodes, input_nodes))

    layer = RbfLayer(input_nodes, layer_nodes, centers, spread=1.5)
    layer.beta = 1.

    inputs = np.array([[-1.], [0], [2.]])

    # How this _should_ work:
    # input -1     layer (0) = (-1 - 0)^2 = 1 
    #
    # input 0.0    layer (0) = (0. - 0.)^2 = 0
    #
    # input 2.     layer (0) = (0. - 2.)^2 = 4

    # Output = exp(-sqrt([ [1], [0], [4] ])**2/(2*1.5))
    # Output = [ [0.716531], [1.], [0.263597]]
    outputs = layer.activate_many(inputs)

    assert np.allclose(outputs, [[0.716531], [1.], [0.263597]], atol=1e-4)


def test_rbf_layer_activates_correctly():
    input_nodes = 3 
    layer_nodes = 2
    
    centers = np.array([0, 1, 2, 3, 4, 5])
    centers = centers.reshape((layer_nodes, input_nodes))

    layer = RbfLayer(input_nodes, layer_nodes, centers, spread=2.)
    layer.beta = 1.

    inputs = np.array([0.5, 1.0, 1.5])

    # How this _should_ work:
    # input 0.5     layer (0, 1, 2) = ((0.5 - 0)^2 + (1 - 1)^2 + (1.5 - 2)^2) = 0.5
    #
    # input 1.0     layer (3, 4, 5) = ((0.5 - 3)^2 + (1 - 4)^2 + (1.5 - 5)^2) = 27.5
    #
    # input 1.5

    # Output = exp(-sqrt([0.5, 27.5])**2/(2*2))
    outputs = tuple(layer.activate(inputs))

    assert np.allclose(outputs, (0.88249, 0.001033), atol=1e-5)

