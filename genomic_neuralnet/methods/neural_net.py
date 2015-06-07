import numpy as np

from genomic_neuralnet.config import MAX_EPOCHS, CONTINUE_EPOCHS 
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet, SupervisedDataSet, UnsupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork

def _get_nn(inputs, hidden):
    # One output layer (1,).
    layers = (inputs,) + hidden + (1,)
    n = buildNetwork(*layers, hiddenclass=SigmoidLayer)
    return n

def get_nn_prediction(train_data, train_truth, test_data, test_truth, hidden=(5,), weight_decay=0.0): 
    mean = np.mean(train_truth)
    sd = np.std(train_truth)
    ds = SupervisedDataSet(len(train_data.columns), 1)
    rows = map(lambda x: x[1], train_data.iterrows())
    for s_data, s_truth in zip(rows, train_truth):
        ds.addSample(s_data, (s_truth - mean) / sd)
    net = _get_nn(len(train_data.columns), hidden)
    trainer = BackpropTrainer(net, ds)
    trainer.trainUntilConvergence(maxEpochs=MAX_EPOCHS, continueEpochs=CONTINUE_EPOCHS, validationProportion=0.25)
    test_ds = UnsupervisedDataSet(len(train_data.columns))
    rows = map(lambda x: x[1], test_data.iterrows())
    for t_data in rows:
        test_ds.addSample(t_data)
    predicted = net.activateOnDataset(test_ds) * sd + mean
    return predicted.ravel()
