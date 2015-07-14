import os
import numpy as np
import pickle
from genomic_neuralnet.neuralnet.layers import LinearLayer 

def test_can_activate_on_multiple_samples_with_complex_shape():
    # Dead simple inputs. The key is that output arrays are correctly shaped.
    input_nodes = 2 
    layer_nodes = 3

    layer = LinearLayer(input_nodes, layer_nodes)
    layer.weights = np.array([1.,2.,3.,4.,5.,6.]).reshape((layer_nodes, input_nodes))
    layer.biases = np.array([1., 2., 3.])

    inputs = np.array([[-1., 0.], [0., 1.], [1., 2.], [2., 3.], [-2, -1]])

    # 2 input values, 3 hidden neurons, 5 samples. Shouldn't get much for broadcasting
    outputs = layer.activate_many(inputs)

    # Should produce 3 values for each of the 5 inputs.
    assert outputs.shape == (5, 3)  

def test_activates_correctly_multiple_inputs():
    input_nodes = 1 
    layer_nodes = 1 
    
    inputs = np.array([[0.5], [1.0], [2.]]) # Nested list.

    layer = LinearLayer(input_nodes, layer_nodes)
    # Force weights + biases to static values for testing purposes.
    layer.weights = np.array([2.]).reshape((layer_nodes, input_nodes))
    layer.biases = np.array([3.])

    # How this _should_ work:
    # input 0.5     layer (2.) + 3. = (2 * 0.5) + 3 = 4.
    #
    # input 1.0     layer (2.) + 3. = (2 * 1.)  + 3 = 5
    #
    # input 2.0     layer (2.) + 3. = (2 * 2.)  + 3 = 7

    # Output = [[4, 5, 7]] 
    outputs = layer.activate_many(inputs)

    assert np.allclose(outputs, [[4.], [5.], [7.]], atol=1e-4)

def test_linear_layer_activates_correctly():
    input_nodes = 3 
    layer_nodes = 2
    
    inputs = np.array([0.5, 1.0, -1.5])

    layer = LinearLayer(input_nodes, layer_nodes)
    # Force weights + biases to static values for testing purposes.
    layer.weights = np.array([1.,2.,3.,4.,5.,6.]).reshape((layer_nodes, input_nodes))
    layer.biases = np.array([1.,2.])

    # How this _should_ work:
    # input 0.5     layer (1, 2, 3) + 1 = (0.5 + 2 - 4.5) + 1 = -1  
    #
    # input 1.0     layer (4, 5, 6) + 2 = (2 + 5 - 9)     + 2 = 0 
    #
    # input 1.5

    # Output = (-1, 0)
    outputs = tuple(layer.activate(inputs))

    assert np.allclose(outputs, (-1., 0.), atol=1e-4)

def test_can_reduce_complex_input_to_single_values():
    input_nodes = 4 
    layer_nodes = 1 
    
    # Create 2 inputs.
    inputs = np.array([-2, -1, 0, 1, 2, 3, 4, 5]).reshape((2, input_nodes))

    layer = LinearLayer(input_nodes, layer_nodes)
    # Force weights + biases to static values for testing purposes.
    layer.weights = np.array([1.,2.,3.,4.]).reshape((layer_nodes, input_nodes))
    layer.biases = np.array([-1.])

    outputs = layer.activate_many(inputs)
    print(outputs)

    assert outputs.shape == (2, 1)
    assert np.allclose(outputs, [[-1], [39]], atol=1e-4)

def test_can_do_interpolation_with_canned_input():
    # The source inputs for the inputs and weight pickles.
    this_dir = os.path.dirname(__file__)
    with open(os.path.join(this_dir, 'inputs.pickle'), 'rb') as f:
        inputs = pickle.load(f)
    with open(os.path.join(this_dir, 'weights.pickle'), 'rb') as f:
        weights = pickle.load(f)
    input_nodes = 1
    layer_nodes = 8
    
    # Apply settings results from file.
    layer = LinearLayer(input_nodes, layer_nodes)
    layer.weights = weights
    layer.biases = np.array([0.])

    outputs = layer.activate_many(inputs)

    assert outputs.shape == (500, 1)
