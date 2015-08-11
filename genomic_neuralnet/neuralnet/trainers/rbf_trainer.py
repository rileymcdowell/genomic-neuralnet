import numpy as np
import numpy.linalg as la
from genomic_neuralnet.neuralnet import RbfLayer, LinearLayer
from genomic_neuralnet.neuralnet.util import orthogonalize_vectors

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
        pseudoinverse solution to the system of linear equations.
        """
        centers = self._rbf_layer.centers

        activations = self._rbf_layer.activate_many(sample_data)
        pseudoinverse = la.pinv(activations)
        weights = np.dot(pseudoinverse, sample_output)

        self._output_layer.weights = weights.T

    def _train_with_best_centers_ols(self, sample_data, sample_output, num_centers=None):
        """
        Select the best centers using the orthogonal least squares algorithm.
        Sample them from the dataset one at a time, always choosing the 
        center that minimizes the remaining MSE.  Continue until num_centers 
        are selected. If num_centers is not specified, we will use (c == n) centers 
        a.k.a len(centers) == len(sample_data). That default is almost certain
        to overfit your data.
        """

        if num_centers is None:
            num_centers = sample_data.shape[0]

        # Start the algorithm by getting all possible pairwise activations.
        all_activations = self._get_all_possible_activations_ols(sample_data)
        orthogonalized_activations = np.copy(all_activations)

        # Iterate and select centers.
        all_activation_idxs = np.arange(all_activations.shape[0])
        available_activation_idxs = np.arange(all_activations.shape[0])
        for center_idx in range(num_centers):
            best_activation_idx = self._select_next_activation_idx( available_activation_idxs
                                                                  , orthogonalized_activations 
                                                                  , sample_output
                                                                  )
            # Remove the activation index that was just chosen.
            chosen_index = available_activation_idxs == best_activation_idx
            available_activation_idxs = available_activation_idxs[~chosen_index]
            # Adjust the remaining activations by removing the projection
            # of the selected activation, leaving only the orthogonal portion. 
            orthogonalize_vectors( all_activations
                                 , orthogonalized_activations
                                 , available_activation_idxs
                                 , best_activation_idx 
                                 )

        # Assign the selected centers to the rbf layer.
        selected_idxs = set(all_activation_idxs) - set(available_activation_idxs)
        selected_centers = sample_data[np.array(list(selected_idxs))]
        self._rbf_layer.centers = selected_centers 

        # Finish training now that the best centers are selected.
        self.train_on_data(sample_data, sample_output)

    def _select_next_activation_idx(self, available_activation_idxs, orthogonalized_activations, sample_output):
        # Select the next orthogonal center by choosing the activation (of a center) that 
        # maximizes the reduction in output energy. This reduces error maximally.
        error_reductions = np.zeros(len(orthogonalized_activations)) # Will be sparse after first selection.
        output_energy = np.dot(sample_output.T, sample_output)
        for activation_idx in list(available_activation_idxs):
            activation = orthogonalized_activations[activation_idx]
            ls_solution = np.dot(activation.T, sample_output) / float(np.dot(activation.T, activation))
            error_reduction = np.dot(ls_solution**2 * activation.T, activation) / output_energy 
            error_reductions[activation_idx] = error_reduction

        largest_error_reduction_idx = np.argmax(error_reductions)

        print('selected center at idx {}'.format(largest_error_reduction_idx))
        return largest_error_reduction_idx 

    def _get_all_possible_activations_ols(self, sample_data):
        self._rbf_layer.num_neurons = 1
        self._rbf_layer.centers = sample_data
        all_activations = []
        for center_idx in range(sample_data.shape[0]):
            self._rbf_layer.centers = np.array([sample_data[center_idx]])
            all_activations.append(self._rbf_layer.activate_many(sample_data))

        all_activations = np.array(all_activations)
        return all_activations.squeeze()

    def train_with_best_centers(self, sample_data, sample_output, num_centers=None, fast_ols=True):
        """
        Select the best centers by sampling them from the dataset one at a
        time, always choosing the center that minimizes the remaining MSE. 
        Continue until num_centers are selected. If num_centers is not 
        specified, we will use (c == n) centers 
        a.k.a len(centers) == len(sample_data). That default is almost certain
        to overfit your data if you have sampling/measurement error.
        """

        if fast_ols:  
            self._train_with_best_centers_ols(sample_data, sample_output, num_centers)
        else:
            self._train_with_best_centers_ls(sample_data, sample_output, num_centers)


    def _train_with_best_centers_ls(self, sample_data, sample_output, num_centers=None):
        """
        Select the best centers by sampling them from the dataset one at a
        time, always choosing the center that minimizes the remaining MSE. 
        Continue until num_centers are selected. If num_centers is not 
        specified, we will use (c == n) centers 
        a.k.a len(centers) == len(sample_data). That default is almost certain
        to overfit your data if you have sampling/measurement error.
        """
        if num_centers is None:
            num_centers = sample_data.shape[0]

        # Start the algorithm with 0 centers chosen. 
        self._rbf_layer.num_neurons = 0 
        self._output_layer.num_neurons = 0
        self._rbf_layer.centers = np.array([]) # Start with nothing
        # Iterate and select centers.
        all_center_idxs = np.arange(sample_data.shape[0])
        available_center_idxs = np.arange(sample_data.shape[0])
        for _ in range(num_centers):
            best_center_idx = self._select_next_best_center(available_center_idxs, sample_data, sample_output)
            # Remove the activation index that was just chosen.
            chosen_index = available_center_idxs == best_center_idx
            available_center_idxs = available_center_idxs[~chosen_index]

        # Finish training now that the best centers are selected.
        self.train_on_data(sample_data, sample_output)

    def _select_next_best_center(self, available_center_idxs, sample_data, sample_output):
        self._rbf_layer.num_neurons += 1
        self._output_layer.num_neurons += 1

        old_centers = self._rbf_layer.centers
        sum_square_error = np.zeros(available_center_idxs.shape)
        for a_idx in range(len(available_center_idxs)):
            center_idx = available_center_idxs[a_idx]
            # Set up the RBF network with the sample center.
            candidate_center = sample_data[center_idx]
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
            sum_square_error[a_idx] = np.sum((activations - sample_output)**2)

        # Find the best starting center and apply it to the network.
        minimum_error_idx = np.argmin(sum_square_error) 
        center_idx = available_center_idxs[minimum_error_idx]
        if old_centers.shape == (0,):
            # Must be assigned, and be a singleton list.
            self._rbf_layer.centers = np.array([sample_data[center_idx]])
        else:
            self._rbf_layer.centers = np.row_stack([old_centers, sample_data[center_idx]])

        print('selected center at idx {}'.format(center_idx))
        return minimum_error_idx
