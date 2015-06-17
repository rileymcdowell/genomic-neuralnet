import numpy as np

from genomic_neuralnet.config import MAX_EPOCHS, CONTINUE_EPOCHS, TRY_CONVERGENCE
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, GaussianLayer, FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet, SupervisedDataSet, UnsupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork

def _get_nn(inputs, hidden):
    """
    Construct a neural network.
    """
    # One output layer (1,).
    layers = (inputs,) + hidden + (1,)
    n = buildNetwork(*layers, hiddenclass=GaussianLayer) # Any need to set sigma here?
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

def get_rbf_nn_prediction(train_data, train_truth, test_data, test_truth, hidden=(5,), weight_decay=0.0): 
    mean = np.mean(train_truth)
    sd = np.std(train_truth)
    ds = SupervisedDataSet(len(train_data.columns), 1)
    rows = map(lambda x: x[1], train_data.iterrows())
    for s_data, s_truth in zip(rows, train_truth):
        ds.addSample(s_data, (s_truth - mean) / sd)
    net = _get_nn(len(train_data.columns), hidden)
    _train_nn(net, ds, weight_decay)
    test_ds = UnsupervisedDataSet(len(train_data.columns))
    rows = map(lambda x: x[1], test_data.iterrows())
    for t_data in rows:
        test_ds.addSample(t_data)
    predicted = net.activateOnDataset(test_ds) * sd + mean
    return predicted.ravel()
