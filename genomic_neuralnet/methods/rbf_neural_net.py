import numpy as np

from sklearn.preprocessing import StandardScaler
from genomic_neuralnet.neuralnet import get_rbf_network
from genomic_neuralnet.neuralnet.trainers import RbfTrainer

def _get_nn(inputs, spread):
    """
    Construct a neural network.
    """
    hidden = None # Start with no centers and train them using LS method.
    ann = get_rbf_network(inputs, hidden, 1, spread=spread)
    return ann

def _train_nn(rbf_net, train_data, train_truth, max_centers):
    """
    A stateful method that trains the network
    on a dataset.
    """
    rbf_trainer = RbfTrainer(rbf_net)
    rbf_trainer.train_with_best_centers(train_data, train_truth[:,np.newaxis], max_centers)

def get_rbf_nn_prediction(train_data, train_truth, test_data, test_truth, max_centers=8, spread=1): 
    scaler = StandardScaler()
    train_truth = scaler.fit_transform(train_truth)
    test_truth = scaler.transform(test_truth)

    net = _get_nn(train_data.shape[1], spread=spread)

    _train_nn(net, train_data, train_truth, max_centers)

    out = net.activate_many(test_data)

    predicted = scaler.inverse_transform(np.array(out))
    return predicted.ravel()
