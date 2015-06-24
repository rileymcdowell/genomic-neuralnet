from __future__ import print_function
import sys
import time
import numpy as np
from genomic_neuralnet.common.base_compare import try_predictors 
from genomic_neuralnet.config import CYCLES, REQUIRED_MARKERS_PROPORTION, CPU_CORES
from genomic_neuralnet.config import markers, pheno
from genomic_neuralnet.config import BACKEND, CELERY_BACKEND, JOBLIB_BACKEND

if BACKEND == CELERY_BACKEND:
    try:
        # Set up celery and define tasks.
        from celery import Celery
        app = Celery('parallel_predictors', backend='redis://localhost', broker='amqp://guest@localhost//')
        celery_try_predictors = app.task(try_predictors)
    except:
        pass

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

    if BACKEND == JOBLIB_BACKEND:
        from joblib import delayed, Parallel
        accuracies = Parallel(n_jobs=CPU_CORES)(delayed(try_predictors)(clean_markers, pheno, prediction_functions) for _ in range(CYCLES))
        accuracies = zip(*accuracies) # Transpose Array, so primary axis is prediction function.
    elif BACKEND == CELERY_BACKEND:
        tasks = [celery_try_predictors.delay(clean_markers, pheno, prediction_functions) for _ in range(CYCLES)]
        while True:
            stati = list(map(lambda x: x.ready(), tasks))
            done = filter(lambda x: x, stati)
            print('Completed {} of {}.'.format(len(done), len(stati)), end='\n')
            if len(stati) == len(done):
                return [t.get() for t in tasks]
            else:
                time.sleep(5)
    else:
        print('Unsupported Processing Backend')
        sys.exit(1)
        
            
    print('')

    return accuracies

