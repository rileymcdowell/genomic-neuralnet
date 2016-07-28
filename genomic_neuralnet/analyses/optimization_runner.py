from __future__ import print_function

import os
import time

import numpy as np
import pandas as pd
import shelve as s

from functools import partial
from itertools import product
from contextlib import closing

from genomic_neuralnet.config import SINGLE_CORE_BACKEND, NUM_FOLDS
from genomic_neuralnet.common import run_predictors
from genomic_neuralnet.util import \
        get_is_on_gpu, get_species_and_trait, get_verbose, get_should_force
from genomic_neuralnet.analyses import OptimizationResult, RUNS

species, trait = get_species_and_trait()
verbose = get_verbose()
on_gpu = get_is_on_gpu()
force = get_should_force()

INDEX_SHELF = 'index.shelf'

def _get_parameter_set(dicts):
    return (dict(zip(dicts, x)) for x in product(*dicts.itervalues()))

def _record_in_master_shelf(method_name, shelf_name):
    shelf_path = os.path.join('shelves', INDEX_SHELF)
    with closing(s.open(shelf_path)) as shelf:
        # The key indicates that this method has been fitted at least one time, and
        # the value is the location of the file where the results are stored.
        shelf[method_name] = shelf_name 
        shelf.close()

def _record_in_model_shelf(shelf_name, df, start, end):
    with closing(s.open(_get_shelf_path(shelf_name))) as shelf:
        result = OptimizationResult(df, end - start, species, trait, NUM_FOLDS, RUNS)
        shelf[_get_shelf_key()] = result

def _is_already_recorded(shelf_name):
    shelf_path = _get_shelf_path(shelf_name)
    shelf_exists = os.path.exists(shelf_path)
    key = _get_shelf_key() 
    if shelf_exists: 
        with closing(s.open(shelf_path)) as shelf:
            if key in shelf:
                return True
    return False # Some prior step was not run. Time to train.

def _get_shelf_key():
    return '|'.join((species, trait, 'gpu' if on_gpu else 'cpu'))

def _get_shelf_path(shelf_name):
    return os.path.join('shelves', shelf_name)

def run_optimization(function, params, shelf_name, method_name, backend=SINGLE_CORE_BACKEND):
    # Check if we even need to do this.
    if _is_already_recorded(shelf_name) and not force:
        print('Training was already completed.')
        print('Run with the "--force" switch to re-train anyway.')
        return 

    start = time.time()
    param_list = list(_get_parameter_set(params))
    df = pd.DataFrame(param_list)
    prediction_functions = map(lambda x: partial(function, **x), param_list)
    accuracies = run_predictors(prediction_functions, backend=backend, runs=RUNS)

    means = []
    std_devs = []
    raw_results = []
    for param_collection, accuracy_arr in zip(param_list, accuracies):
        means.append(np.mean(accuracy_arr))
        # Lose 1 degree of freedom b/c we estimated mean also.
        std_devs.append(np.std(accuracy_arr, ddof=1))         
        raw_results.append(accuracy_arr)

    df['mean'] = means
    df['std_dev'] = std_devs
    df['raw_results'] = raw_results # Stores one list per single datagrid cell.

    if verbose:
        print(df)
        print()

    end = time.time()

    print("Recording fitting results to shelf '{}'.".format(shelf_name))
    print()

    _record_in_master_shelf(method_name, shelf_name)
    _record_in_model_shelf(shelf_name, df, start, end)

    print('Best Parameters Were:')
    max_mean = np.argmax(df['mean'])
    print(df.iloc[max_mean])
    print()

    print('Fitting Parameters Were:')
    print('Runs: {}'.format(RUNS))
    print('Folds: {}'.format(NUM_FOLDS))
    print('Raw Results Shape: {}'.format(np.array(raw_results).shape))
    print()

    print('Done.')


