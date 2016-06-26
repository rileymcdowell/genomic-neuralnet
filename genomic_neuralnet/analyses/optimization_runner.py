from __future__ import print_function
import os

import numpy as np
import pandas as pd
import shelve as s

from functools import partial
from genomic_neuralnet.config import SINGLE_CORE_BACKEND
from genomic_neuralnet.common import run_predictors

from genomic_neuralnet.util import get_species_and_trait, get_verbose

speces, trait = get_species_and_trait()
verbose = get_verbose()

from itertools import product

def _get_parameter_set(dicts):
    return (dict(zip(dicts, x)) for x in product(*dicts.itervalues()))

def run_optimization(function, params, shelf_name, backend=SINGLE_CORE_BACKEND):
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

    print("Writing output to shelf '{}'.".format(shelf_name))

    shelf_path = os.path.join('plots', shelf_name)
    shelf = s.open(shelf_path)
    shelf['|'.join(('species', 'trait'))] = df
    shelf.close()

    print('Best Parameters Were:')

    max_mean = np.argmax(df['mean'])
    print(df.iloc[max_mean])

    print('Done.')


