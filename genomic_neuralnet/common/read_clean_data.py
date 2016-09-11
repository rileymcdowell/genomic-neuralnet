from __future__ import print_function 

import numpy as np

from genomic_neuralnet.config import REQUIRED_MARKER_CALL_PROPORTION, \
                                     REQUIRED_MARKERS_PER_SAMPLE_PROP
from genomic_neuralnet.util import get_markers_and_pheno

def get_clean_data(species, trait):
    markers, pheno = get_markers_and_pheno(species, trait)

    # Remove missing phenotypic values from both datasets.
    has_trait_data = pheno.notnull()
    clean_pheno = pheno[has_trait_data].copy(deep=True)
    clean_markers = markers.drop(markers.columns[~has_trait_data], axis=1)

    # Remove samples with many missing marker calls.
    sample_missing_count = clean_markers.isnull().sum()
    num_markers = len(clean_markers)
    max_missing_allowed = 1. - REQUIRED_MARKERS_PER_SAMPLE_PROP
    required_markers = int(np.ceil(num_markers * max_missing_allowed))
    bad_samples = (sample_missing_count > (num_markers * max_missing_allowed))
    clean_markers = clean_markers.drop(clean_markers.columns[bad_samples], axis=1)
    clean_pheno = clean_pheno[~bad_samples]
    
    # Remove markers with many missing values calls.
    marker_missing_count = clean_markers.T.isnull().sum()
    num_samples = len(clean_markers.columns)
    max_missing_allowed = 1. - REQUIRED_MARKER_CALL_PROPORTION
    required_samples = int(np.ceil(num_samples * max_missing_allowed))
    bad_markers = (marker_missing_count > (num_samples * max_missing_allowed))
    clean_markers = clean_markers[~bad_markers]

    # Impute missing values with the mean for that column.
    clean_markers = clean_markers.fillna(clean_markers.mean())

    # Reset all indices to avoid future indexing loc/iloc confusion.
    clean_pheno = clean_pheno.reset_index(drop=True)
    clean_markers = clean_markers.reset_index(drop=True)

    clean_pheno = clean_pheno.copy(deep=True)
    clean_markers = clean_markers.copy(deep=True)

    return clean_markers, clean_pheno

