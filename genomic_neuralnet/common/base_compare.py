from __future__ import print_function
import sys
import numpy as np
import scipy.stats as sps

from genomic_neuralnet.config import TRAIN_SIZE, TRAIT_NAME, NUM_FOLDS

def try_predictor(markers, pheno, prediction_function, random_seed, id_val=None):
    """
    Pass in markers, phenotypes, and a list of prediction functions.
    Returns the prediction accuracy (pearson r) relative to measured phenotype. 
    """
    # Re-seed the generator.
    np.random.seed(random_seed)
     
    trait = TRAIT_NAME

    # Grab the fold to be left out for this iteration.
    fold_idx, _ = id_val 

    # Pre-Filter data that is missing phenotypic measurements for
    # this trait.
    good_idxs = pheno.ix[~pheno[trait].isnull()].index.values

    # Randomize the order of the index data (in-place).
    np.random.shuffle(good_idxs)

    # Build the folds of the dataset. 
    folds = np.array(np.array_split(good_idxs, NUM_FOLDS))

    # Partition the indexes into in and out of sample sections.
    out_of_sample_idxs = folds[fold_idx]
    mask = np.ones(len(folds), dtype=bool)
    mask[fold_idx] = 0
    in_sample_idxs = np.hstack(folds[mask])
    
    # Train data
    train_data = markers.ix[:,in_sample_idxs].T.values
    train_truth = pheno[trait].ix[in_sample_idxs].values

    # Test data
    test_data = markers.ix[:,out_of_sample_idxs].T.values
    test_truth = pheno[trait].ix[out_of_sample_idxs].values
     
    predicted = prediction_function(train_data, train_truth, test_data, test_truth)
    accuracy = sps.stats.pearsonr(predicted, test_truth)[0]
    
    return accuracy, id_val

