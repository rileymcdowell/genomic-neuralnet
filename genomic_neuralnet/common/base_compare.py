from __future__ import print_function
import sys
import numpy as np
import scipy.stats as sps

from genomic_neuralnet.config import NUM_FOLDS
from genomic_neuralnet.common.read_clean_data import get_clean_data

def try_predictor(prediction_function, random_seed, species, trait, id_val=None, retry_nans=False, config_gpu=False):
    """
    Pass in markers, phenotypes, and a list of prediction functions.
    Returns the prediction accuracy (pearson r) relative to measured phenotype. 
    """
    markers, pheno = get_clean_data(species, trait)

    # Force set GPU just before fitting if requested.
    # This option overrides any environment variable.
    if config_gpu:
        import theano.sandbox.cuda
        if not theano.sandbox.cuda.cuda_enabled:
            theano.sandbox.cuda.use('gpu')

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
     
    while True:
        predicted = prediction_function(train_data, train_truth, test_data, test_truth)
        accuracy = sps.stats.pearsonr(predicted, test_truth)[0]

        if np.isnan(accuracy) and retry_nans:
            continue
        else:
            return accuracy, id_val

