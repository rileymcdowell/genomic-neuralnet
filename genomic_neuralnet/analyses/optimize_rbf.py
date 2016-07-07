from __future__ import print_function
import sys
import pandas as pd
import numpy as np

from functools import partial

from genomic_neuralnet.config import JOBLIB_BACKEND
from genomic_neuralnet.common import run_predictors
from genomic_neuralnet.methods import get_rbf_nn_prediction 
from genomic_neuralnet.analyses import run_optimization

def main():
    # These ranges likely contain the global maximum. Find it. 
    center = [50, 100, 150, 200, 250]
    spread = np.arange(50, 1001, 50) 
    params = {'centers': center, 'spread': spread}

    run_optimization(get_rbf_nn_prediction, params, 'optimal_rbf.shelf', 'RBF')

if __name__ == '__main__':
    main()

