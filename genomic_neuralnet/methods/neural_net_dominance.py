import numpy as np

from genomic_neuralnet.config import MAX_EPOCHS, CONTINUE_EPOCHS, TRY_CONVERGENCE, USE_ARAC
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet, SupervisedDataSet, UnsupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork

def _get_nn(inputs, hidden):
    """
    Construct a neural network.
    """
    # One output layer (1,).
    layers = (inputs,) + hidden + (1,)
    n = buildNetwork(*layers, hiddenclass=SigmoidLayer, fast=USE_ARAC)
    return n

def _train_nn(neuralnet, training_set, weight_decay):
    """
    A stateful method that trains the network
    on a dataset.
    """
    trainer = BackpropTrainer(neuralnet, training_set, weightdecay=weight_decay, momentum=0.5)
    if TRY_CONVERGENCE: # Try to converge to an optimal solution.
        trainer.trainUntilConvergence(maxEpochs=MAX_EPOCHS, continueEpochs=CONTINUE_EPOCHS, validationProportion=0.25)
    else: # Train a specific number of epochs and then stop.
        trainer.trainEpochs(epochs=MAX_EPOCHS)

def _convert_to_individual_alleles(array):
    """
    Convert SNPs to individual copies so neuralnet can learn dominance relationships.
    [-1, 0, 1] => [(0, 0), (0, 1), (1, 1)] => [0, 0, 0, 1, 1, 1]
    """
    array = array.values # We don't want a pandas series anymore.
    # Set non-integer values to 0 (het)
    array = np.trunc(array)
    incr = array + 1 # Now we have 0, 1, and 2
    incr = incr[:,:,np.newaxis] # Add another dimension.
    pairs = np.pad(incr, ((0,0), (0,0), (0,1)), mode='constant') # Append one extra 0 value to final axis.
    twos = np.sum(pairs, axis=2) == 2
    pairs[twos] = [1,1]
    x, y, z = pairs.shape
    pairs = pairs.reshape((x, y*z)) # Merge pairs to one axis.
    return pairs

def get_nn_dom_prediction(train_data, train_truth, test_data, test_truth, hidden=(5,), weight_decay=0.0): 
    # Convert data to capture dominance.
    train_data, test_data = tuple(map(_convert_to_individual_alleles, [train_data, test_data]))

    mean = np.mean(train_truth)
    sd = np.std(train_truth)

    # Supervised training dataset.
    ds = SupervisedDataSet(train_data.shape[1], 1)
    ds.setField('input', train_data) 
    ds.setField('target', (train_truth[:, np.newaxis] - mean) / sd)

    net = _get_nn(train_data.shape[1], hidden)

    _train_nn(net, ds, weight_decay)

    # Unsupervised (test) dataset.
    test_ds = UnsupervisedDataSet(test_data.shape[1])
    test_ds.setField('sample', test_data)

    predicted = net.activateOnDataset(test_ds) * sd + mean
    return predicted.ravel()
