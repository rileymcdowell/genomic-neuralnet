from __future__ import print_function
import numpy as np

from genomic_neuralnet.config import JOBLIB_BACKEND
from genomic_neuralnet.methods import get_lasso_prediction 
from genomic_neuralnet.analyses import run_optimization

def main():
    alphas = list(np.arange(0.01, 1.01,0.01))
    params = {'alpha': alphas}
    run_optimization(get_lasso_prediction, params, 'optimal_lasso.shelf', backend=JOBLIB_BACKEND)

if __name__ == '__main__':
    main()

