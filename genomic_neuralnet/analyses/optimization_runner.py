from __future__ import print_function

import os
import time

import numpy as np
import pandas as pd
import shelve as s

from functools import partial
from genomic_neuralnet.config import SINGLE_CORE_BACKEND
from genomic_neuralnet.common import run_predictors
from genomic_neuralnet.util import \
        get_is_on_gpu, get_species_and_trait, get_verbose
from genomic_neuralnet.analyses import OptimizationResult

species, trait = get_species_and_trait()
verbose = get_verbose()

from itertools import product

def _get_parameter_set(dicts):
    return (dict(zip(dicts, x)) for x in product(*dicts.itervalues()))

DIRECTORY_SHELF = 'directory.shelf'

def _record_in_master_shelf(method_name, shelf_name):
    shelf_path = os.path.join('shelves', DIRECTORY_SHELF)
    shelf = s.open(shelf_path)
    # The key indicates that this method has been fitted at least one time, and
    # the value is the location where the results are stored.
    key = method_name 
    value = shelf_name
    shelf[key] = value 
    shelf.close()

def run_optimization(function, params, shelf_name, method_name, backend=SINGLE_CORE_BACKEND):
    start = time.time()
    param_list = list(_get_parameter_set(params))
    df = pd.DataFrame(param_list)
    prediction_functions = map(lambda x: partial(function, **x), param_list)
    accuracies = run_predictors(prediction_functions, backend=backend)

    means = []
    std_devs = []
    for param_collection, accuracy_arr in zip(param_list, accuracies):
        means.append(np.mean(accuracy_arr))
        std_devs.append(np.std(accuracy_arr))

    df['mean'] = means
    df['std_dev'] = std_devs

    if verbose:
        print(df)

    print('Done')
    end = time.time()

    print("Recording fitting results to shelf '{}'.".format(shelf_name))

    _record_in_master_shelf(method_name, shelf_name)

    shelf_path = os.path.join('shelves', shelf_name)
    shelf = s.open(shelf_path)
    key = '|'.join((species, trait))
    result = OptimizationResult(df, end - start, species, trait)
    shelf[key] = result
    shelf.close()

    print('Best Parameters Were:')

    max_mean = np.argmax(df['mean'])
    print(df.iloc[max_mean])

    print('Done.')


