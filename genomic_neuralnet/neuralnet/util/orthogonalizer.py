from __future__ import print_function
import numpy as np

def orthogonalize_vectors( original_vectors
                         , orthogonalized_vectors
                         , unselected_idxs 
                         , last_selected_idx 
                         ):
    """
    A function used to orthogonalize a list of vectors relative
    to one another by iteratively selecting vectors and subtracting their
    projection from the remaining, unselected vectors. 

    This procedure has utility for the Orthogonal Least Squares RBF network
    training algorithm.

    original_vectors - an unmodified copy of the set of vectors to orthogonalize.
    orthogonalized_vectors - a set of vectors that is mutated to be more orthogonal.
    unselected_idxs - the indexes of the vectors that have not yet been selected.
    last_selected_idx - the index of the vector whose projection should be
        subtracted subtract from the rest of the vectors.
    """
    # Calculate the projection of the selected vector onto the remaining ones.
    chosen_vector = orthogonalized_vectors[last_selected_idx]
    remaining_vectors = original_vectors[unselected_idxs]
    numerators = np.dot(chosen_vector.T, remaining_vectors.T)
    denominator = np.dot(chosen_vector.T, chosen_vector)
    projection_coefficients = numerators / float(denominator)
    # Remove the projection onto the remaining vectors.
    remaining_orthogonalized_vectors = orthogonalized_vectors[unselected_idxs]
    projection_to_subtract = np.outer(chosen_vector, projection_coefficients).T
    orthogonalized_vectors[unselected_idxs] = (remaining_orthogonalized_vectors - projection_to_subtract)

