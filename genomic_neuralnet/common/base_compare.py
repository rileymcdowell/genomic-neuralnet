from __future__ import print_function
import sys
import numpy as np
import scipy.stats as sps

from genomic_neuralnet.config import TRAIN_SIZE, NUM_FOLDS

def try_predictor(markers, pheno, prediction_function, random_seed, id_val=None):
    """
    Pass in markers, phenotypes, and a list of prediction functions.
    Returns the prediction accuracy (pearson r) relative to measured phenotype. 
    """

    # Re-seed the generator.
    np.random.seed(random_seed)
     
    # Grab the fold to be left out for this iteration.
    fold_idx, _ = id_val 

    # Grab indexes of all of the phenotypic measurements.
    indexes = pheno.index.values.copy()

    # Randomize the order of the index data (in-place).
    np.random.shuffle(indexes)

    # Build the folds of the dataset. 
    folds = np.array(np.array_split(indexes, NUM_FOLDS))

    # Partition the indexes into in and out of sample sections.
    out_of_sample_idxs = folds[fold_idx]
    mask = np.ones(len(folds), dtype=bool)
    mask[fold_idx] = 0
    in_sample_idxs = np.hstack(folds[mask])

    # Train data
    train_data = markers.iloc[:,in_sample_idxs].T.values
    train_truth = pheno.iloc[in_sample_idxs].values

    # Test data
    test_data = markers.iloc[:,out_of_sample_idxs].T.values
    test_truth = pheno.iloc[out_of_sample_idxs].values

    # Make the actual prediction random again by re-seeding the generator.
    np.random.seed()
     
    predicted = prediction_function(train_data, train_truth, test_data, test_truth)
    accuracy = sps.stats.pearsonr(predicted, test_truth)[0]
    
    return accuracy, id_val

