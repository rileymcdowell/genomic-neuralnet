import numpy as np
from genomic_neuralnet.neuralnet.util import orthogonalize_vectors

def test_does_not_mutate_first_selected_vector():
    """
    A pair of vectors requires the calculation of a single
    projection coefficient of the first vector onto the
    second vector. This is the simplest scenario to test.
    """

    original_vectors = np.array([[2.,5.],[8.,4.]])
    orthogonalized_vectors = np.copy(original_vectors)

    # Imagine that we selected the first vector, and want to orthogonalize
    # the other one.
    orthogonalize_vectors(original_vectors, orthogonalized_vectors, np.array([1]), 0)

    # First vector unchanged.
    assert np.allclose(orthogonalized_vectors[0], original_vectors[0])

def test_resulting_vectors_are_orthogonal():
    """
    A pair of vectors requires the calculation of a single
    projection coefficient of the first vector onto the
    second vector. This is the simplest scenario to test.
    """

    original_vectors = np.array([[2.,5.],[8.,4.]])
    orthogonalized_vectors = np.copy(original_vectors)

    # Imagine that we selected the first vector, and want to orthogonalize
    # the other one.
    orthogonalize_vectors(original_vectors, orthogonalized_vectors, np.array([1]), 0)

    # Dotted orthogonal vectors only have a diagonal component.
    dotted = np.dot(orthogonalized_vectors, orthogonalized_vectors.T)
    assert np.sum(dotted) - np.trace(dotted) < 0.0001

def test_larger_vector_collection_is_made_orthogonal():
    """
    A pair of vectors requires the calculation of a single
    projection coefficient of the first vector onto the
    second vector. This is the simplest scenario to test.
    """

    original_vectors = np.array([[2.,5.,3], [8.,4.,2], [-5,7,9], [-2,-2,-4]])
    orthogonalized_vectors = np.copy(original_vectors)

    # Imagine that we selected the first vector, and want to orthogonalize
    # the other one.

    orthogonalize_vectors(original_vectors, orthogonalized_vectors, np.array([1,2,3]), 0)
    orthogonalize_vectors(original_vectors, orthogonalized_vectors, np.array([2,3]), 1)
    orthogonalize_vectors(original_vectors, orthogonalized_vectors, np.array([3]), 2)

    # Dotted orthogonal vectors only have a diagonal component.
    dotted = np.dot(orthogonalized_vectors, orthogonalized_vectors.T)
    assert np.sum(dotted) - np.trace(dotted) < 0.0001

