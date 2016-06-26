from __future__ import print_function

import os
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.optimizers import Adagrad
from keras.regularizers import WeightRegularizer

from sklearn.preprocessing import MinMaxScaler 

class NeuralNetContainer(object):
    def __init__(self): 
        self.model = None
        self.learning_rate = None
        self.weight_decay = 0.0
        self.dropout_prob = 0.0
        self.epochs = 25
        self.batch_size = 100
        self.hidden_layers = (10,)

def _build_nn(net_container, n_features):
    model = Sequential() 

    # Change scale from (-1, 1) to (0, 1)
    model.add(Lambda(lambda x: (x + 1) / 2, input_shape=(n_features,), output_shape=(n_features,)))

    if net_container.weight_decay > 0.0:
        weight_regularizer = WeightRegularizer(net_container.weight_decay)
    else: 
        weight_regularizer = None 

    last_dim = n_features
    for lidx, n_nodes in enumerate(net_container.hidden_layers):
        # Layer, activation, and dropout, in that order.
        model.add(Dense(output_dim=n_nodes, input_dim=last_dim, W_regularizer=weight_regularizer))
        model.add(Activation('sigmoid'))
        if net_container.dropout_prob > 0.0:
            model.add(Dropout(net_container.dropout_prob))
        last_dim = n_nodes

    model.add(Dense(output_dim=1, input_dim=last_dim, bias=False))
    model.add(Activation('linear'))

    if not net_container.learning_rate is None:
        optimizer = Adagrad(lr=net_container.learning_rate)
    else:
        optimizer = Adagrad()

    model.compile( optimizer=optimizer
                 , loss='mean_squared_error'
                 , metrics=['accuracy']
                 )

    net_container.model = model

def _train_net(container, X, y):
    """ 
    Given a container, X (inputs), and y (outputs) train the network in the container. 
    """
    model = container.model
    epochs = container.epochs
    batch_size = container.batch_size
    verbose = int(container.verbose)

    model.fit(X, y, nb_epoch=epochs, batch_size=batch_size, verbose=verbose)

def _predict(container, X):
    model = container.model

    return model.predict(X)

def get_net_prediction( train_data, train_truth, test_data, test_truth
                      , hidden=(5,), weight_decay=0.0, dropout_prob=0.0
                      , learning_rate=None, epochs=25, batch_size=100
                      , verbose=False
                      ):

    container = NeuralNetContainer()
    container.learning_rate = learning_rate
    container.dropout_prob = dropout_prob
    container.weight_decay = weight_decay
    container.epochs = epochs
    container.batch_size = batch_size
    container.hidden_layers = hidden
    container.verbose = verbose

    mms = MinMaxScaler(feature_range= (-1, 1)) # Scale output from -1 to 1.
    train_y = mms.fit_transform(train_truth[:,np.newaxis])

    n_features = train_data.shape[1]

    # Build the network. 
    _build_nn(container, n_features)

    # Train the network.
    _train_net( container, train_data, train_y)

    # Unsupervised (test) dataset.
    predicted = _predict(container, test_data)
    predicted = mms.inverse_transform(predicted)
    
    return predicted.ravel()

