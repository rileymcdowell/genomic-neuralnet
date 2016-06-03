from __future__ import print_function

import numpy as np
import os
import glob
import parse

import pandas as pd

_this_dir = os.path.dirname(__file__)
SNP_DATA_FILE = os.path.join(_this_dir, 'Snp_Data.csv')
PHENOTYPIC_DATA_FILES = glob.glob(os.path.join(_this_dir, 'Phenotypic_Data Folder', 'DATA_*.csv'))

def main():
    geno_df = pd.read_csv(SNP_DATA_FILE, index_col='Genotype')
    geno_df = geno_df.astype(float) 
    geno_df = geno_df.replace(to_replace=-9, value=np.NaN)
    geno_df = geno_df - 1 # Change to scale [-1, 1]
    geno_df.to_csv('genotypes.csv', index=None, header=False, na_rep='NaN', float_format='%1.0f')

    last_pheno_df = None
    for filepath in PHENOTYPIC_DATA_FILES:
        filename = os.path.split(filepath)[1]
        phenotype_name = parse.parse('DATA_{}.csv', filename).fixed[0]

        pheno_df = pd.read_csv(filepath, index_col='Genotype')
        pheno_df = pheno_df[['Derregressed_BV']] # Just the deregressed breeding value.
        pheno_df.columns = [phenotype_name]

        if last_pheno_df is None:
            last_pheno_df = pheno_df
        else:
            last_pheno_df = last_pheno_df.join(pheno_df)


    last_pheno_df.to_csv('phenotypes.csv', na_rep='NaN')

if __name__ == '__main__':
    main()
