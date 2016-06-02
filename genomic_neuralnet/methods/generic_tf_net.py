from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

from genomic_neuralnet.util import NeuralnetConfig
from sklearn.preprocessing import MinMaxScaler 

_this_dir = os.path.dirname(__file__)
LOG_DIR = os.path.join(_this_dir, '..', '..', 'tf_log_dir')

EPSILON = 1e-4

class NeuralNetContainer(object):
    def __init__(self): 
        self.output_func = None
        self.train_func = None
        self.session = None
        self.x_var = None
        self.truth_var = None
        self.error_func = None
        self.weight_decay = None
        self.keep_prob = None
        self.writer = None
        self.learning_rate = None

def _random_matrix(shape, name=None):
    v = tf.Variable(tf.random_normal(shape, mean=0, stddev=0.5), name=name)
    return v

def _build_nn(net_config, n_features):
    session = tf.Session()

    x = tf.placeholder(tf.float32, shape=(None, n_features))
    truth = tf.placeholder(tf.float32, shape=(None, 1))

    # Parameters.
    weight_decay = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)

    last_output = x
    last_n_nodes = n_features
    all_weights = []
    for lidx, n_nodes in enumerate(net_config.hidden_layers):
        with tf.name_scope('Layer{}'.format(lidx)):
            w = _random_matrix((last_n_nodes, n_nodes), name='W{}'.format(lidx))
            b = _random_matrix((n_nodes,), name='B{}'.format(lidx))

            # Remember the weights for weight decay.
            weight_count = tf.reduce_prod(tf.shape(w), keep_dims=True)
            all_weights.append(tf.reshape(w, weight_count))

            inputs = tf.nn.bias_add(tf.matmul(last_output, w), b)
            activations = tf.mul(tf.sub(tf.sigmoid(inputs), 0.5), 2)

            # Apply dropout.
            activations = tf.nn.dropout(activations, keep_prob)
            
            last_n_nodes = n_nodes
            last_output = activations

    with tf.name_scope('Output'):
        w = _random_matrix((last_n_nodes, 1), name='OutputWeights')
        weight_count = tf.reduce_prod(tf.shape(w), keep_dims=True)
        all_weights.append(tf.reshape(w, weight_count))

        # Output layer doesn't have bias node.
        inputs = tf.matmul(last_output, w)
        #activations = tf.clip_by_value(inputs, -1, 1) # Linear output layer, clipped.
        activations = inputs

    final_out = activations 

    network_error = tf.reduce_mean(tf.squared_difference(truth, final_out))

    all_weights = tf.concat(0, all_weights)
    weight_error = tf.nn.l2_loss(all_weights)

    # Error term is network error plus weight error.
    error = tf.add(network_error , tf.mul(weight_decay, weight_error))

    trainer = tf.train.AdagradOptimizer(learning_rate)
    train_func = trainer.minimize(error)

    session.run(tf.initialize_all_variables())

    # Write the graph.
    writer = tf.train.SummaryWriter(LOG_DIR, graph=session.graph)

    # Build a container to save the useful parts of the network.
    container = NeuralNetContainer()
    container.writer = writer
    container.session = session
    container.output_func = final_out
    container.train_func = train_func
    container.x_var = x
    container.truth_var = truth
    container.network_error = network_error
    container.error_func = error
    container.weight_decay = weight_decay
    container.keep_prob = keep_prob
    container.learning_rate = learning_rate

    return container 

def _train_net(container, net_config, X, y, weight_decay_lambda, dropout_keep_prob):
    """ 
    Given a container, X (inputs), and y (outputs) train the network in the container. 

    Parameters:
        weight_decay_lambda - lambda in the weight decay function.
        dropout_keep_prob - the probability of keeping a neuron in a training epoch.

    """
    sess = container.session
    train_func = container.train_func
    x = container.x_var
    truth = container.truth_var
    output = container.output_func
    error = container.error_func
    network_error = container.network_error
    weight_decay = container.weight_decay
    keep_prob = container.keep_prob
    learning_rate = container.learning_rate

    current_learning_rate = net_config.initial_learning_rate 
    min_error = np.inf
    iters_without_decrease = 0
    last_error = min_error
    for epoch_idx in range(0, net_config.max_epochs):

        # By doing random mini-batching, turn this into stochastic gradient descent.
        sample_idxs = np.random.choice(X.shape[0], size=X.shape[0], replace=False)
        for batch_idxs in np.array_split(sample_idxs, net_config.batch_splits):
            x_sample = X[batch_idxs]
            y_sample = y[batch_idxs]

            feed_dict = { x: x_sample
                        , truth: y_sample
                        , weight_decay: weight_decay_lambda
                        , keep_prob: dropout_keep_prob
                        , learning_rate: current_learning_rate
                        }
            sess.run(train_func, feed_dict=feed_dict)

        # Calculate end-of-epoch error.
        err = sess.run(error, feed_dict)
        feed_dict = { x: X
                    , truth: y
                    , keep_prob: 1.0
                    , weight_decay: weight_decay_lambda
                    }

        # Maybe report Network MSE.
        if epoch_idx % net_config.report_every == 0:
            print('Epoch {}. Network MSE = {}'.format(epoch_idx, err))
            print(current_learning_rate)

        if net_config.active_learning_rate:
            # Update the learning rate.
            if err < last_error:
                current_learning_rate *= net_config.learning_rate_multiplier 
            else:
                new_rate = current_learning_rate / net_config.learning_rate_divisor
                current_learning_rate = np.max([new_rate, EPSILON])

        # Reset error counter.
        last_error = err

        if not net_config.try_convergence:
            continue
        else:
            # If we're going to try to converge, we need to
            # evaluate the current error including weight decay.
            feed_dict[weight_decay] = weight_decay_lambda
            err = sess.run(error, feed_dict)

        if err < min_error:
            min_error = err
            iters_without_decrease = 0
        else:
            iters_without_decrease += 1

        if iters_without_decrease > net_config.continue_epochs:
            err = sess.run(error, feed_dict)
            print('Convergence threshold reached. Training stopped.')
            break# Break out early if it seems like we've converged.

def _predict(container, X):
    sess = container.session
    keep_prob = container.keep_prob
    x = container.x_var
    output = container.output_func

    feed_dict = { x: X
                , keep_prob:1.0
                }
    output = sess.run(output, feed_dict=feed_dict)

    return output

def get_net_prediction( train_data, train_truth, test_data, test_truth
                      , net_config = None, weight_decay=0.0, dropout_keep_prob=1.0): 
    """ Defaults to a fully-connected network """

    if net_config is None:
        net_config = NeuralnetConfig()

    mms = MinMaxScaler(feature_range= (-1, 1)) # Scale output from -1 to 1.
    train_y = mms.fit_transform(train_truth[:,np.newaxis])

    n_features = train_data.shape[1]

    # Build the network. 
    net_container = _build_nn(net_config, n_features)

    # Train the network.
    _train_net( net_container, net_config, train_data, train_y
              , weight_decay, dropout_keep_prob)

    # Unsupervised (test) dataset.
    predicted = _predict(net_container, test_data)
    predicted = mms.inverse_transform(predicted)
    
    return predicted.ravel()

