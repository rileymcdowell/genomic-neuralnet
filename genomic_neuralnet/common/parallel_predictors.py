import numpy as np
from joblib import delayed, Parallel
from genomic_neuralnet.common.base_compare import try_predictors 
from genomic_neuralnet.config import CYCLES, REQUIRED_MARKERS_PROPORTION, CPU_CORES
from genomic_neuralnet.config import markers, pheno

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

    accuracies = Parallel(n_jobs=CPU_CORES)(delayed(try_predictors)(clean_markers, pheno, prediction_functions) for _ in range(CYCLES))
    accuracies = zip(*accuracies) # Transpose Array, so primary axis is prediction function.

    return accuracies

