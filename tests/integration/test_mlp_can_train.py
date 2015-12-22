from __future__ import print_function
import numpy as np

from genomic_neuralnet.neuralnet import get_mlp_network_regressor, get_mlp_network_classifier 
from genomic_neuralnet.neuralnet.trainers import MlpTrainer 

def test_can_execute_forward_pass_one():
    """
    Following example on 'https://www4.rgu.ac.uk/files/chapter3 - bp.pdf'
    This article is linked from the wikipedia backpropagation page as a resource for
    an in-depth explanation of the backpropagation algorithm (as of 9-13-15).

    This article does not include neuron bias, so we will turn it off for this test.
    """
    net = get_mlp_network_classifier(2, 2, 1, apply_bias=False)
    net.layers[0].weights = np.array([[0.1, 0.8],[0.4, 0.6]])
    net.layers[1].weights = np.array([[0.3, 0.9]])

    input = np.array([0.35, 0.9])
    layer_io = net.activate_include_layer_inputs(input)
    ((_, hidden), (_, output)) = layer_io

    expected_hidden = np.array([0.680267, 0.663738])
    expected_output = np.array([0.690258])

    assert np.allclose(expected_hidden, hidden, atol=0.0001)
    assert np.allclose(expected_output, output, atol=0.0001)

def test_can_execute_forward_pass_two():
    """
    Following example on 'https://www4.rgu.ac.uk/files/chapter3 - bp.pdf'
    This article is linked from the wikipedia backpropagation page as a resource for
    an in-depth explanation of the backpropagation algorithm (as of 9-13-15).

    This article does not include neuron bias, so we will turn it off for this test.
    """
    net = get_mlp_network_classifier(2, 2, 1, apply_bias=False)
    net.layers[0].weights = np.array([[0.1, 0.5],[0.3, 0.2]])
    net.layers[1].weights = np.array([[0.2, 0.1]])

    input = np.array([0.1, 0.7])
    layer_io = net.activate_include_layer_inputs(input)
    ((_, hidden), (_, output)) = layer_io

    expected_hidden = np.array([0.589040, 0.542398])
    expected_output = np.array([0.542906])

    assert np.allclose(expected_hidden, hidden, atol=0.0001)
    assert np.allclose(expected_output, output, atol=0.0001)

def test_can_execute_forward_pass_432():
    """
    Trying a 4-3-2 network to confirm broadcasting rules are working.
    """
    net = get_mlp_network_classifier(4, 3, 2, apply_bias=False)
    net.layers[0].weights = np.array([[0.1, 0.4, 0.7, 0.1],[0.2, 0.5, 0.8, 0.2], [0.3, 0.6, 0.9, 0.3]])
    net.layers[1].weights = np.array([[0.4, 0.6, 0.8], [0.5, 0.7, 0.9]])

    input = np.array([0.1, 0.2, 0.3, 0.4])
    layer_io = net.activate_include_layer_inputs(input)
    ((_, hidden), (_, output)) = layer_io

    expected_hidden = np.array([0.584191, 0.608259, 0.631812])
    expected_output = np.array([0.751024, 0.783555])

    assert np.allclose(expected_hidden, hidden, atol=0.00001)
    assert np.allclose(expected_output, output, atol=0.00001)

def test_can_execute_single_backpropagation_pass_one():
    """
    Following example on 'https://www4.rgu.ac.uk/files/chapter3 - bp.pdf'
    This article is linked from the wikipedia backpropagation page as a resource for
    an in-depth explanation of the backpropagation algorithm (as of 9-13-15).

    This article does not include neuron bias, so we will turn it off for this test.
    """
    net = get_mlp_network_classifier(2, 2, 1, apply_bias=False)
    net.layers[0].weights = np.array([[0.1, 0.8],[0.4, 0.6]])
    net.layers[1].weights = np.array([[0.3, 0.9]])

    input = np.array([0.35, 0.9])
    desired_output = np.array([0.5])
    trainer = MlpTrainer(net)

    #trainer.train_on_data(input[np.newaxis,:], desired_output[np.newaxis,:]) 
    trainer.train_on_datum(input, desired_output) 

    expected_hidden_weights = np.array([[0.0992, 0.7978], [0.3972, 0.5928]])
    expected_output_weights = np.array([0.2724, 0.8731]) 

    assert np.allclose(net.layers[0].weights, expected_hidden_weights, atol=0.0001)
    assert np.allclose(net.layers[1].weights, expected_output_weights, atol=0.0001)
        
def test_can_execute_single_backpropagation_pass_two():
    """
    Following example on 'https://www4.rgu.ac.uk/files/chapter3 - bp.pdf'
    This article is linked from the wikipedia backpropagation page as a resource for
    an in-depth explanation of the backpropagation algorithm (as of 9-13-15).

    This article does not include neuron bias, so we will turn it off for this test.
    """
    net = get_mlp_network_classifier(2, 2, 1, apply_bias=False)
    net.layers[0].weights = np.array([[0.1, 0.5],[0.3, 0.2]])
    net.layers[1].weights = np.array([[0.2, 0.1]])

    input = np.array([0.1, 0.7])
    desired_output = np.array([1.0])
    trainer = MlpTrainer(net)

    #trainer.train_on_data(input[np.newaxis,:], desired_output[np.newaxis,:]) 
    trainer.train_on_datum(input, desired_output)

    expected_hidden_weights = np.array([[0.1007, 0.5051], [0.3005, 0.2032]])
    expected_output_weights = np.array([0.2668, 0.1615])

    assert np.allclose(net.layers[0].weights, expected_hidden_weights, atol=0.0001)
    assert np.allclose(net.layers[1].weights, expected_output_weights, atol=0.0001)

def test_can_single_backpropagation_pass_432():
    """
    Trying a 4-3-2 network to confirm broadcasting rules are working.
    """
    net = get_mlp_network_classifier(4, 3, 2, apply_bias=False)
    net.layers[0].weights = np.array([[0.1, 0.4, 0.7, 0.1],[0.2, 0.5, 0.8, 0.2], [0.3, 0.6, 0.9, 0.3]])
    net.layers[1].weights = np.array([[0.4, 0.6, 0.8], [0.5, 0.7, 0.9]])

    input = np.array([0.1, 0.2, 0.3, 0.4])
    desired_output = np.array([0.1, 0.0])
    trainer = MlpTrainer(net)

    trainer.train_on_datum(input, desired_output)

    expected_hidden_weights = np.array([ [0.097664, 0.395328, 0.692992, 0.090656]
                                       , [0.196514, 0.493028, 0.789541, 0.186055]
                                       , [0.295430, 0.590859, 0.886289, 0.281719] ])
    expected_output_weights = np.array([ [0.328885, 0.525955, 0.723088]
                                       , [0.422368, 0.619170, 0.816040]
                                       ])
    print(expected_hidden_weights)
    print(net.layers[0].weights)

    assert np.allclose(net.layers[0].weights, expected_hidden_weights, atol=0.0001)
    assert np.allclose(net.layers[1].weights, expected_output_weights, atol=0.0001)

def test_backpropagation_decreases_error():
    """
    Following example on 'https://www4.rgu.ac.uk/files/chapter3 - bp.pdf'
    This freely available book chapter is linked from the wikipedia backpropagation 
    page as a resource for an in-depth explanation of the backpropagation 
    algorithm (as of 9-13-15).

    This chapter does not include neuron bias, so we will turn it off for this test.
    """
    net = get_mlp_network_classifier(2, 2, 1, apply_bias=False)
    net.layers[0].weights = np.array([[0.1, 0.8],[0.4, 0.6]])
    net.layers[1].weights = np.array([[0.3, 0.9]])

    input = np.array([0.35, 0.9])
    desired_output = np.array([0.5])
    trainer = MlpTrainer(net)

    original_error = desired_output - net.activate(input)
    trainer.train_on_data(input[np.newaxis,:], desired_output[np.newaxis,:]) 
    new_error = desired_output - net.activate(input)

    assert original_error < new_error
    assert np.allclose(original_error, -0.19, atol=0.001)
    assert np.allclose(new_error, -0.18205, atol=0.0001)

def test_mlp_classifier_can_approximate_xor_function():
    """
    This is the hello world of using sigmoid neuron networks
    for classification. We should be able to nail this one.
    """
    x = np.array([[1,1], [0,1], [1,0], [0,0]])
    y = np.array([[0],[1],[1],[0]])

    mlp_network = get_mlp_network_classifier(2, 2, 1, apply_bias=False)
    mlp_trainer = MlpTrainer(mlp_network, learning_rate=0.9)

    print('before')
    print(mlp_network.layers[0].weights)
    print(mlp_network.layers[1].weights)
    print()
    import time
    time.sleep(1)

    for _ in range(100000000):
        predicted_out = mlp_network.activate_many(x)
        original_error = y - predicted_out 
        #mlp_trainer.train_on_data(x, y)
        for i in range(4):
            mlp_trainer.train_on_datum(x[i], y[i])
        new_error = y - mlp_network.activate_many(x)
        total_error = np.sum(np.square(original_error)) 
        #print('total error', np.sum(np.square(original_error)))
        if total_error < 0.020:
            break
        #print()

    print('after')
    print(mlp_network.layers[0].weights)
    print(mlp_network.layers[1].weights)
    print()


    predicted_out = mlp_network.activate_many(x)
    print(x)
    print(y)
    print(predicted_out)
    print('took {} iterations to learn xor'.format(_))
    # Should be no problem getting them all right with high certainty.
    assert np.allclose(y, predicted_out, rtol=0.0, atol=2e-1) 
    
if __name__ == '__main__':
    test_mlp_classifier_can_approximate_xor_function()
