from __future__ import print_function

import numpy as np
import tensorflow as tf

from genomic_neuralnet.util import NeuralnetConfig
from sklearn.preprocessing import MinMaxScaler 

class DropoutNetContainer(object):
    def __init__(self): 
        self.output_func = None
        self.train_func = None
        self.session = None
        self.x_var = None
        self.truth_var = None
        self.error_func = None
        self.keep_prob = None # Dropout node keep probability.

def _random_matrix(shape):
    v = tf.Variable(tf.random_normal(shape, mean=0, stddev=1))
    return v

def _build_nn(net_config, n_features):
    session = tf.Session()

    x = tf.placeholder(tf.float32, shape=(None, n_features))
    truth = tf.placeholder(tf.float32, shape=(None, 1))
    keep_prob = tf.placeholder(tf.float32)

    layer_nodes = net_config.hidden_layers + (1,) # The output layer is a layer also.

    last_output = x
    last_n_nodes = n_features
    for n_nodes in layer_nodes:
        w = _random_matrix((last_n_nodes, n_nodes))
        b = _random_matrix((n_nodes,))

        inputs = tf.matmul(last_output, w) + b
        activations = tf.sigmoid(inputs)
        
        activations = tf.nn.dropout(activations, keep_prob)

        last_n_nodes = n_nodes
        last_output = activations

    final_out = last_output

    error = tf.reduce_sum(tf.squared_difference(truth, last_output))

    trainer = tf.train.RMSPropOptimizer(net_config.learning_rate)
    train_func = trainer.minimize(error)

    session.run(tf.initialize_all_variables())

    # Build a container to save the useful parts of the network.
    container = DropoutNetContainer()
    container.session = session
    container.output_func = final_out
    container.train_func = train_func
    container.x_var = x
    container.truth_var = truth
    container.error_func = error
    container.keep_prob = keep_prob

    return container 

def _train_net(container, net_config, X, y, dropout_keep_prob):
    """ 
    Given a container, X (inputs), and y (outputs) train the network in the container. 
    """
    sess = container.session
    train_func = container.train_func
    x = container.x_var
    truth = container.truth_var
    output = container.output_func
    error = container.error_func
    keep_prob = container.keep_prob

    min_error = np.inf
    iters_without_decrease = 0
    for epoch_idx in range(0, net_config.max_epochs):

        # By doing random mini-batching, turn this into stochastic gradient descent.
        sample_idxs = np.random.choice(X.shape[0], size=X.shape[0], replace=False)
        for batch_idxs in np.array_split(sample_idxs, net_config.batch_splits):
            x_sample = X[batch_idxs]
            y_sample = y[batch_idxs]

            feed_dict = {x: x_sample, truth: y_sample, keep_prob:dropout_keep_prob}
            sess.run(train_func, feed_dict=feed_dict)

        feed_dict = {x: X, truth: y, keep_prob:1.0}
        if epoch_idx % 100 == 0:
            err = sess.run(error, feed_dict)
            print('Epoch {}. Sum Squared Error = {}'.format(epoch_idx, err))

        if not net_config.try_convergence:
            continue
        else:
            # If we're going to try to converge, we need to
            # evaluate the current error.
            err = sess.run(error, feed_dict)

        if err < min_error:
            min_error = err
            iters_without_decrease = 0
        else:
            iters_without_decrease += 1

        if iters_without_decrease > net_config.continue_epochs:
            err = sess.run(error, feed_dict)
            print('Convergence threshold reached. Training stopped.')
            return # Break out early if it seems like we've converged.

def _predict(container, X):
    sess = container.session
    train_func = container.train_func
    x = container.x_var
    truth = container.truth_var
    output = container.output_func
    error = container.error_func
    keep_prob = container.keep_prob

    feed_dict = {x: X, keep_prob:1.0}
    output = sess.run(output, feed_dict=feed_dict)

    return output

def get_do_net_prediction(train_data, train_truth, test_data, test_truth, net_config = None, dropout_keep_prob=0.75): 

    if net_config is None:
        net_config = NeuralnetConfig()

    mms = MinMaxScaler() # Scale output to 0-1.
    train_y = mms.fit_transform(train_truth[:,np.newaxis])

    n_features = train_data.shape[1]

    # Build the network. 
    net_container = _build_nn(net_config, n_features)

    # Train the network.
    _train_net(net_container, net_config, train_data, train_y, dropout_keep_prob)

    # Unsupervised (test) dataset.
    predicted = _predict(net_container, test_data)
    predicted = mms.inverse_transform(predicted)

    return predicted.ravel()

def main():
    x = np.arange(-5, 5)[:,np.newaxis]
    y = np.sin(x)


    get_do_net_prediction(x, y)


if __name__ == '__main__':
    main()
