from __future__ import print_function
import sys
import numpy as np
import scipy.stats as sps

from genomic_neuralnet.config import TRAIN_SIZE, TRAIT_NAME

def try_predictors(markers, pheno, prediction_functions, print_progress=True):
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
    train_data = markers.ix[:,train_idxs].T
    train_truth = pheno[trait].ix[train_idxs]

    # Test data
    test_data = markers.ix[:,test_idxs].T
    test_truth = pheno[trait].ix[test_idxs]
     
    accuracies = []
    for prediction_function in prediction_functions:
        try:
            predicted = prediction_function(train_data, train_truth, test_data, test_truth)
            accuracies.append(sps.stats.pearsonr(predicted, test_truth)[0])
        except:
            print(train_data)
            print(train_truth)
            sys.exit()

    # Print dots to show progress.
    if print_progress:
        print('.', end='')
        sys.stdout.flush()

    return tuple(accuracies) 

