from __future__ import print_function

import numpy as np
import os
import bz2file as bz2 # Required to decompress compressd files with multiple streams.
import glob
import parse

import pandas as pd

from StringIO import StringIO

_this_dir = os.path.dirname(__file__)
GENOTYPIC_DATA_FILE = os.path.join(_this_dir, 'genotypes.txt.bz2')
PHENOTYPIC_DATA_FILE = os.path.join(_this_dir, 'Phenotypic_Data Folder', 'DATA_*.csv')

def main():
    # Decompress bzip2 file.
    if not os.path.exists(os.path.join(_this_dir, 'genotypes.csv.bz2')):
        with bz2.BZ2File(GENOTYPIC_DATA_FILE, 'rb') as f:
            geno_df = pd.read_csv(f, index_col='ID')
        geno_df = geno_df - 1 # Change to scale [-1, 1]
        geno_df.to_csv('genotypes.csv.bz2', index=None, header=False, compression='bz2')

    if not os.path.exists(os.path.join(_this_dir, 'phenotypes.csv')):
        pheno_df = pd.read_csv(PHENOTYPIC_DATA_FILE, index_col='ID')
        pheno_df = pheno_df.replace('.', np.NaN)
        pheno_df = pheno_df.astype(float)
        pheno_df.to_csv('phenotypes.csv', na_rep='NaN')

if __name__ == '__main__':
    main()
