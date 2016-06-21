from __future__ import print_function
import numpy as np

from functools import partial
from genomic_neuralnet.config import JOBLIB_BACKEND
from genomic_neuralnet.methods import get_en_prediction
from genomic_neuralnet.analyses import run_optimization

def main():
    alphas = list(np.arange(0.05, 1.00, 0.05))
    l1_ratios = list(np.arange(0.05, 1.00, 0.05))
    params = {'alpha': alphas, 'l1_ratio': l1_ratios}
    run_optimization(get_en_prediction, params, 'optimal_en.shelf', backend=JOBLIB_BACKEND)

if __name__ == '__main__':
    main()

