from __future__ import print_function
import sys
import time
import numpy as np
import itertools
from genomic_neuralnet.common.base_compare import try_predictor
from genomic_neuralnet.config import CYCLES, REQUIRED_MARKERS_PROPORTION, CPU_CORES
from genomic_neuralnet.config import markers, pheno
from genomic_neuralnet.config import BACKEND, CELERY_BACKEND, JOBLIB_BACKEND

if BACKEND == CELERY_BACKEND:
    try:
        # Set up celery and define tasks.
        from celery import Celery
        app = Celery('parallel_predictors', backend='redis://localhost', broker='amqp://guest@localhost//')
        celery_try_predictor = app.task(try_predictor)
    except:
        pass

def _run_joblib(job_params):
    from joblib import delayed, Parallel
    accuracies = Parallel(n_jobs=CPU_CORES)(delayed(try_predictor)(*x) for x in job_params) 
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

def run_predictors(prediction_functions):
    """
    Runs all prediction functions on the same data in a 
    batch process across the configured number of CPUs. 
    Returns the accuracies of the functions as list of arrays
    ordered by function.
    """

    # Remove markers with many missing values by filtering on the NOT of the columns with too many nulls.
    clean_markers = markers.ix[~(markers.T.isnull().sum() > len(markers) * (1 - REQUIRED_MARKERS_PROPORTION))]
    # Impute missing values with the mean for that column
    clean_markers = markers.fillna(markers.mean())

    # Set up the parameters for processing.
    pf_idxs = range(len(prediction_functions))
    job_params = [(clean_markers, pheno, prediction_functions[pf_idx], (idx, pf_idx)) for idx in range(CYCLES) for pf_idx in pf_idxs] 

    if BACKEND == JOBLIB_BACKEND:
        accuracies = _run_joblib(job_params)
    elif BACKEND == CELERY_BACKEND:
        accuracies = _run_celery(job_params)
    else:
        print('Unsupported Processing Backend')
        sys.exit(1)
        
    accuracies.sort(key=lambda x: x[1][1]) # Sort by prediction function
    grouped = [accuracies[idx:idx+CYCLES] for idx in range(0, len(accuracies), CYCLES)] # Group by prediction function
    return map(lambda x: map(lambda y: y[0], x), grouped) # Just return the accuracies
    
    return accuracies

