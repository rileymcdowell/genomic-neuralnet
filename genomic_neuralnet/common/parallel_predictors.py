from __future__ import print_function
import sys
import time
import numpy as np

from itertools import chain

from genomic_neuralnet.common.base_compare import try_predictor
from genomic_neuralnet.config import REQUIRED_MARKER_CALL_PROPORTION, \
                                     REQUIRED_MARKERS_PER_SAMPLE_PROP
from genomic_neuralnet.config import CPU_CORES, NUM_FOLDS
from genomic_neuralnet.config import CELERY_BACKEND, JOBLIB_BACKEND, \
                                     SINGLE_CORE_BACKEND, INIT_CELERY
from genomic_neuralnet.util import get_markers_and_pheno

ACCURACY_IDX = 0
IDENTIFIER_IDX = 1
FOLD_IDX = 0
PRED_FUNC_IDX = 1

if INIT_CELERY == CELERY_BACKEND:
    try:
        # Set up celery and define tasks.
        from celery import Celery
        name = 'parallel_predictors'
        backend = 'redis://localhost'
        broker = 'amqp://guest@localhost//'
        app = Celery(name, backend=backend, broker=broker)
        celery_try_predictor = app.task(try_predictor)
    except:
        pass

def _run_joblib(job_params):
    from joblib import delayed, Parallel
    accuracies = Parallel(n_jobs=CPU_CORES)(delayed(try_predictor)(*x) for x in job_params)
    return accuracies

def _run_debug(job_params):
    """ Single process for easy debugging. """
    accuracies = []
    for args in job_params:
        accuracies.append(try_predictor(*args))
    return accuracies

def _run_celery(job_params):
    tasks = [celery_try_predictor.delay(*x) for x in job_params]
    while True:
        stati = list(map(lambda x: x.ready(), tasks))
        done = filter(lambda x: x, stati)
        print('Completed {} of {} cycles.'.format(len(done), len(stati)), end='\n')
        if len(stati) == len(done):
            break
        else:
            time.sleep(15)
    print('')
    accuracies = [t.get() for t in tasks]
    return accuracies

def _get_clean_data():
    markers, pheno = get_markers_and_pheno()

    # Remove missing phenotypic values from both datasets.
    has_trait_data = pheno.notnull()
    clean_pheno = pheno[has_trait_data].copy(deep=True)
    clean_markers = markers.drop(markers.columns[~has_trait_data], axis=1)

    # Remove samples with many missing marker calls.
    sample_missing_count = clean_markers.isnull().sum()
    num_markers = len(clean_markers)
    max_missing_allowed = 1. - REQUIRED_MARKERS_PER_SAMPLE_PROP
    required_markers = int(np.ceil(num_markers * max_missing_allowed))
    bad_samples = (sample_missing_count > (num_markers * max_missing_allowed))
    clean_markers = clean_markers.drop(clean_markers.columns[bad_samples], axis=1)
    clean_pheno = clean_pheno[~bad_samples]
    
    # Remove markers with many missing values calls.
    marker_missing_count = clean_markers.T.isnull().sum()
    num_samples = len(clean_markers.columns)
    max_missing_allowed = 1. - REQUIRED_MARKER_CALL_PROPORTION
    required_samples = int(np.ceil(num_samples * max_missing_allowed))
    bad_markers = (marker_missing_count > (num_samples * max_missing_allowed))
    clean_markers = clean_markers[~bad_markers]

    # Impute missing values with the mean for that column.
    clean_markers = clean_markers.fillna(clean_markers.mean())

    # Reset all indices to avoid future indexing loc/iloc confusion.
    clean_pheno = clean_pheno.reset_index(drop=True)
    clean_markers = clean_markers.reset_index(drop=True)

    return clean_pheno, clean_markers

def run_predictors(prediction_functions, backend=SINGLE_CORE_BACKEND, random_seed=1, runs=1, retry_nans=False):
    """
    Runs all prediction functions on the same data in a 
    batch process across the configured number of CPUs. 
    Returns the accuracies of the functions as list of arrays
    ordered by function.
    """
    clean_pheno, clean_markers = _get_clean_data()

    # Set up the parameters for processing.
    pred_func_idxs = range(len(prediction_functions))
    accuracy_results = []
    for _ in range(runs):
        job_params = []
        for prediction_function_idx in pred_func_idxs:
            for fold_idx in range(NUM_FOLDS):
                identifier = (fold_idx, prediction_function_idx)
                prediction_function = prediction_functions[prediction_function_idx]
                params = ( clean_markers, clean_pheno, prediction_function
                         , random_seed, identifier, retry_nans)
                job_params.append(params)

        # Run the jobs and return a tuple of the accuracy and the id (which is also a tuple).
        if backend == JOBLIB_BACKEND:
            accuracies = _run_joblib(job_params)
        elif backend == CELERY_BACKEND:
            accuracies = _run_celery(job_params)
        elif backend == SINGLE_CORE_BACKEND:
            accuracies = _run_debug(job_params)
        else:
            print('Unsupported Processing Backend')
            sys.exit(1)
        accuracy_results.append(accuracies)
        random_seed += 1 # Increment seed to obtain new data folds this run.

    accuracies = list(chain.from_iterable(accuracy_results))

    # Sort results by prediction function, default is ascending.
    # This puts things back into the order they were made
    # which is also the order they were passed into this function.
    accuracies.sort(key=lambda x: x[IDENTIFIER_IDX][PRED_FUNC_IDX])

    grouped = []
    # Create groups of results, one group per prediction function.
    # Because we just sorted the results, new groups start every 
    # (NUM_FOLDS * runs) elements.

    group_size = NUM_FOLDS * runs
    for idx in range(0, len(accuracies), group_size):
        group = accuracies[idx:idx+(group_size)]
        grouped.append(group)

    # Drop everything from the output except the accuracy, but
    # still return the accuracies grouped by which prediction
    # function ran them.
    return map(lambda x: map(lambda y: y[ACCURACY_IDX], x), grouped)     

