import numpy as np
from numpy.linalg import pinv
from genomic_neuralnet.neuralnet import RbfLayer, LinearLayer

class RbfTrainer(object):
    def __init__(self, rbf_network):
        # Check that this is compatible with pseudoinverse training.
        assert len(rbf_network.layers) == 2
        assert type(rbf_network.layers[0]) == RbfLayer 
        assert type(rbf_network.layers[1]) == LinearLayer 
        self._target_network = rbf_network 
        self._rbf_layer = self._target_network.layers[0]
        self._output_layer = self._target_network.layers[1]
        self._output_layer.biases = 0 # Don't use these.

    def train_on_data(self, sample_data, sample_output):
        """
        This method will train an RBF network assuming that centers
        have already been set. The resulting weights in the output layer are set
        to minimize the ordinary least squares output error using the
        pseudoinverse weight-setting method.
        """
        centers = self._rbf_layer.centers

        activations = self._rbf_layer.activate_many(sample_data)
        pseudoinverse = pinv(activations)
        weights = np.dot(pseudoinverse, sample_output)

        self._output_layer.weights = weights.T

    def train_with_best_centers(self, sample_data, sample_output, max_centers=None):
        """
        Select the best centers by sampling them from the dataset one at a
        time, always choosing the center that minimizes the remaining MSE. 
        Continue until max_centers are selected. If max_centers is not 
        specified, we will use (c == n) centers 
        a.k.a len(centers) == len(sample_data). That default is almost certain
        to overfit your data if you have sampling/measurement error.
        """
        if max_centers is None:
            max_centers = sample_data.shape[0]

        # Start the algorithm with 0 centers chosen. 
        self._rbf_layer.num_neurons = 0 
        self._output_layer.num_neurons = 0
        self._rbf_layer.centers = np.array([]) # Start with nothing
        # Iterate and select centers.
        available_centers = list(sample_data)
        for _ in range(max_centers):
            self._select_next_best_center(available_centers, sample_data, sample_output)

        # Finish training now that the best centers are selected.
        self.train_on_data(sample_data, sample_output)

    def _select_next_best_center(self, available_centers, sample_data, sample_output):
        self._rbf_layer.num_neurons += 1
        self._output_layer.num_neurons += 1

        old_centers = self._rbf_layer.centers
        sum_square_error = np.zeros((len(available_centers),))
        for idx in range(len(available_centers)):
            # Set up the RBF network with the sample center.
            candidate_center = available_centers[idx]
            if old_centers.shape == (0,):
                # Must be assigned, and be a singleton list.
                self._rbf_layer.centers = np.array([candidate_center])
            else:
                # Stack additional centers onto the existing ones.
                self._rbf_layer.centers = np.row_stack([old_centers, candidate_center])

            # Train the network.
            self.train_on_data(sample_data, sample_output)

            # Record the sum squared error.
            activations = self._target_network.activate_many(sample_data)
            sum_square_error[idx] = np.sum((activations - sample_output)**2)

        # Find the best starting center and apply it to the network.
        minimum_error = np.argmin(sum_square_error)
        if old_centers.shape == (0,):
            # Must be assigned, and be a singleton list.
            self._rbf_layer.centers = np.array([available_centers[minimum_error]])
        else:
            self._rbf_layer.centers = np.row_stack([old_centers, available_centers[minimum_error]])

        del available_centers[minimum_error]
        print('selected center at {}'.format(sample_data[minimum_error]))
        
