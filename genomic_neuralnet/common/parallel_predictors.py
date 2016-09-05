from __future__ import print_function 

import sys
import time
import numpy as np

from itertools import chain

from genomic_neuralnet.common.base_compare import try_predictor
from genomic_neuralnet.config import REQUIRED_MARKER_CALL_PROPORTION, \
                                     REQUIRED_MARKERS_PER_SAMPLE_PROP
from genomic_neuralnet.config import CPU_CORES, NUM_FOLDS
from genomic_neuralnet.config import PARALLEL_BACKEND, SINGLE_CORE_BACKEND
from genomic_neuralnet.util import get_markers_and_pheno, get_use_celery, \
                                   get_verbose, get_reuse_celery_cache, \
                                   get_celery_gpu
from genomic_neuralnet.common.read_clean_data import get_clean_data

ACCURACY_IDX = 0
IDENTIFIER_IDX = 1
FOLD_IDX = 0
PRED_FUNC_IDX = 1

MAX_INT = 2**31 * 2 - 1
MIN_INT = 0

def _dot_wrapper(func, *params):
    res = func(*params)
    print('.', end='')
    sys.stdout.flush()
    return res

def _run_joblib(job_params):
    from joblib import delayed, Parallel

    accuracies = Parallel(n_jobs=CPU_CORES)(delayed(_dot_wrapper)(try_predictor, *x) for x in job_params)
    return accuracies

def _run_debug(job_params):
    """ Single process for easy debugging. """
    accuracies = []
    for args in job_params:
        accuracies.append(_dot_wrapper(try_predictor, *args))
    return accuracies

def _run_celery(job_params):
    from genomic_neuralnet.common.celery_slave \
        import celery_try_predictor, get_num_workers, get_queue_length, \
               disk_cache, load_and_clear_cache, is_disk_cached
               
    job_idx = 0
    results = {} 
    done = 0
    while True:
        queue_len = get_queue_length()
        workers = get_num_workers()
        # Keep putting messages on the queue until there
        # is one message waiting for every worker.
        desired_messages = workers
        # Account for exhausting the work queue.
        remaining_jobs = len(job_params) - job_idx
        num_to_add = np.min([desired_messages - queue_len, remaining_jobs])
        if get_verbose():
            print('{} Workers / Desired Messages'.format(workers))
            print('{} In Flight'.format(len(results)))
            print('{} Completed'.format(done))
            print('{} Not Started'.format(len(job_params) - len(results) - done))
            print('Adding {} messages'.format(num_to_add))
        # Add messages to fill queue.
        for _ in range(num_to_add):
            if get_reuse_celery_cache():
                while is_disk_cached(job_idx):
                    print('Skipping {}. Already completed'.format(job_idx))
                    done += 1
                    job_idx += 1
                    # Skip what's already done.
            if job_idx >= len(job_params):
                continue # Don't run past the end of the list of parameters to run.
            else:
                delayed = celery_try_predictor.delay(*job_params[job_idx])
                results[job_idx] = delayed
                job_idx += 1

        # Cache finished work.
        keys = results.keys()
        for key in keys:
            result = results[key]
            if result.ready():
                accs = result.get()
                disk_cache(accs, key)
                del results[key] # Stop tracking.
                if get_verbose():
                    print('Done with id {}'.format(key))
                done += 1
        if done == len(job_params):
            if get_verbose():
                print('All done!')
            break # All done!
        else:
            # Wait a bit while work gets done.
            print('Completed {} of {} cycles.'.format(done, len(job_params)))
            time.sleep(10) # One check every ten seconds is plenty.

    accuracies = load_and_clear_cache(range(len(job_params)))
    return accuracies

def run_predictors(prediction_functions, backend=SINGLE_CORE_BACKEND, random_seed=0, runs=1, retry_nans=False):
    """
    Runs all prediction functions on the same data in a 
    batch process across the configured number of CPUs. 
    Returns the accuracies of the functions as list of arrays
    ordered by function.
    """
    # Set up the parameters for processing.
    pred_func_idxs = range(len(prediction_functions))
    accuracy_results = []
    for _ in range(runs):
        job_params = []
        for prediction_function_idx in pred_func_idxs:
            for fold_idx in range(NUM_FOLDS):
                identifier = (fold_idx, prediction_function_idx)
                prediction_function = prediction_functions[prediction_function_idx]
                params = (prediction_function, random_seed, identifier, retry_nans, get_celery_gpu())
                job_params.append(params)

        # Run the jobs and return a tuple of the accuracy and the id (which is also a tuple).
        if backend == PARALLEL_BACKEND and get_use_celery():
            accuracies = _run_celery(job_params)
        elif backend == PARALLEL_BACKEND:
            accuracies = _run_joblib(job_params)
        elif backend == SINGLE_CORE_BACKEND:
            accuracies = _run_debug(job_params)
        else:
            print('Unsupported Backend Settings.')
            sys.exit(1)
        accuracy_results.append(accuracies)
        random_seed = np.random.randint(MIN_INT, MAX_INT) # New seed to obtain new data folds this run.

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
