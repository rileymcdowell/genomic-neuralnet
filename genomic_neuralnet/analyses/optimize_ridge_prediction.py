from __future__ import print_function

import shelve as s
import numpy as np
import pandas as pd

from functools import partial
from genomic_neuralnet.methods import get_rr_prediction
from genomic_neuralnet.analyses import run_optimization

def main():
    # alphas = list(np.logspace(-8., 8., base=10, num=17)) # Wide search
    alphas =  list(np.linspace(10, 1000, num=100)) # Narrow Search
    params = {'alpha': alphas}
    run_optimization(get_rr_prediction, params, 'optimal_rr.shelf', 'RR') # Already parallel.

if __name__ == '__main__':
    main()

