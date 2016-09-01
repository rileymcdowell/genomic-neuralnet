import numpy as np

from sklearn.preprocessing import StandardScaler
from simplenet import get_rbf_network
from simplenet.trainers import RbfTrainer

def _get_nn(inputs, spread):
    """
    Construct a neural network.
    """
    hidden = None # Start with no centers and train them using LS method.
    ann = get_rbf_network(inputs, hidden, 1, spread=spread)
    return ann

def _train_nn(rbf_net, train_data, train_truth, centers):
    """
    A stateful method that trains the network
    on a dataset.
    """
    rbf_trainer = RbfTrainer(rbf_net)
    rbf_trainer.train_with_best_centers(train_data, train_truth[:,np.newaxis], centers)

def get_rbf_nn_prediction(train_data, train_truth, test_data, test_truth, centers=8, spread=1, iter_id=0): 
    train_truth = train_truth[:,np.newaxis]
    test_truth = test_truth[:,np.newaxis]

    scaler = StandardScaler()
    train_truth = scaler.fit_transform(train_truth).ravel()
    test_truth = scaler.transform(test_truth).ravel()

    net = _get_nn(train_data.shape[1], spread=spread)

    _train_nn(net, train_data, train_truth, centers)

    out = net.activate_many(test_data)

    predicted = scaler.inverse_transform(np.array(out))
    return predicted.ravel()
