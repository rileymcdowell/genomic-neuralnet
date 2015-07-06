from __future__ import print_function
import sys
import numpy as np
import scipy.stats as sps

from genomic_neuralnet.config import TRAIN_SIZE, TRAIT_NAME

def try_predictor(markers, pheno, prediction_function, id_val=None):
    """
    Pass in markers, phenotypes, and a list of prediction functions.
    Returns the prediction accuracy (pearson r) relative to measured phenotype. 
    """
    # Re-seed the generator.
    np.random.seed()
     
    trait = TRAIT_NAME

    good_idxs = pheno.ix[~pheno[trait].isnull()].index.values
    train_idxs = np.random.choice(good_idxs, size=round(len(good_idxs) * TRAIN_SIZE), replace=False)
    test_idxs = np.setdiff1d(good_idxs, train_idxs)

    # Train data
    train_data = markers.ix[:,train_idxs].T.values
    train_truth = pheno[trait].ix[train_idxs].values

    # Test data
    test_data = markers.ix[:,test_idxs].T.values
    test_truth = pheno[trait].ix[test_idxs].values
    
     
    predicted = prediction_function(train_data, train_truth, test_data, test_truth)
    accuracy = sps.stats.pearsonr(predicted, test_truth)[0]
    
    return accuracy, id_val

