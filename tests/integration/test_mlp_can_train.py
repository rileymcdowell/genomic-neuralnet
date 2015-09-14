from __future__ import print_function
import numpy as np

from genomic_neuralnet.neuralnet import get_mlp_network_regressor, get_mlp_network_classifier 
from genomic_neuralnet.neuralnet.trainers import MlpTrainer 

def test_can_execute_forward_pass():
    """
    Following example on 'https://www4.rgu.ac.uk/files/chapter3 - bp.pdf'
    This article is linked from the wikipedia backpropagation page as a resource for
    an in-depth explanation of the backpropagation algorithm (as of 10-13-15).

    This article does not include neuron bias, so we will turn it off for this test.
    """
    net = get_mlp_network_classifier(2, 2, 1, apply_bias=False)
    net.layers[0].weights = np.array([[0.1, 0.8],[0.4, 0.6]])
    net.layers[1].weights = np.array([0.3, 0.9])

    output = net.activate(np.array([0.35, 0.9]))
        
    assert np.allclose(0.69, output, atol=0.01)

def test_can_execute_single_backpropagation_pass():
    """
    Following example on 'https://www4.rgu.ac.uk/files/chapter3 - bp.pdf'
    This article is linked from the wikipedia backpropagation page as a resource for
    an in-depth explanation of the backpropagation algorithm (as of 10-13-15).

    This article does not include neuron bias, so we will turn it off for this test.
    """
    net = get_mlp_network_classifier(2, 2, 1, apply_bias=False)
    net.layers[0].weights = np.array([[0.1, 0.8],[0.4, 0.6]])
    net.layers[1].weights = np.array([0.3, 0.9])

    input = np.array([0.35, 0.9])
    desired_output = np.array([0.5])
    trainer = MlpTrainer(net)

    trainer.train_on_datum(input, desired_output) 

    assert np.allclose(net.layers[0].weights, np.array([[0.10, 0.80], [0.40, 0.59]]), atol=0.01)
    assert np.allclose(net.layers[1].weights, np.array([0.27, 0.87]), atol=0.01)
        

def test_backpropagation_decreases_error():
    """
    Following example on 'https://www4.rgu.ac.uk/files/chapter3 - bp.pdf'
    This freely available book chapter is linked from the wikipedia backpropagation 
    page as a resource for an in-depth explanation of the backpropagation 
    algorithm (as of 10-13-15).

    This chapter does not include neuron bias, so we will turn it off for this test.
    """
    net = get_mlp_network_classifier(2, 2, 1, apply_bias=False)
    net.layers[0].weights = np.array([[0.1, 0.8],[0.4, 0.6]])
    net.layers[1].weights = np.array([0.3, 0.9])

    input = np.array([0.35, 0.9])
    desired_output = np.array([0.5])
    trainer = MlpTrainer(net)

    original_error = desired_output - net.activate(input)
    trainer.train_on_datum(input, desired_output) 
    new_error = desired_output - net.activate(input)

    assert original_error < new_error
    assert np.allclose(original_error, -0.19, atol=0.01)
    assert np.allclose(new_error, -0.18, atol=0.01)

#def test_mlp_can_approximate_sin_function():
#    x = np.mgrid[-np.pi:np.pi:50j][:, np.newaxis]
#    x_dense = np.mgrid[-np.pi:np.pi:500j][:, np.newaxis]
#    def y_func(x):
#        return (np.sin(x) + 1) / 2 # Scale to [0,1].
#    y = y_func(x)
#    y_dense = y_func(x_dense)
#
#    hidden_neurons = 8
#
#    mlp_network = get_mlp_network_regressor(1, hidden_neurons, 1)
#    mlp_trainer = mlpTrainer(mlp_network)
#
#    mlp_trainer.train_on_data(x, y)
#
#    predicted_out = mlp_network.activate_many(x_dense)
#
#    # Should be no problem getting within 0.1 of actual function. 
#    assert np.allclose(y_dense, predicted_out, rtol=0.0, atol=1e-1) 
#     
