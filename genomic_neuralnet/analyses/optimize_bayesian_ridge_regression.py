from __future__ import print_function

from genomic_neuralnet.config import JOBLIB_BACKEND
from genomic_neuralnet.methods import get_brr_prediction
from genomic_neuralnet.analyses import run_optimization

def main():
    nums = (1e15, 1e10, 1e5, 1, 0, -1, -10, -100, -1000, -10000)
    alpha_1 = nums
    alpha_2 = nums
    lambda_1 = nums
    lambda_2 = nums 
    params = { 'alpha_1': alpha_1, 'alpha_2': alpha_2
             , 'lambda_1': lambda_1, 'lambda_2': lambda_2
             }
    run_optimization(get_brr_prediction, params, 'optimal_en.shelf', 'BRR', backend=JOBLIB_BACKEND)

if __name__ == '__main__':
    main()

