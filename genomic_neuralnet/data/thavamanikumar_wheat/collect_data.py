from __future__ import print_function
"""
From Thavamanikumar et al. 2015:

FileS1 is data for 1975 SNPS from CxH
FileS2 is data for 1483 SNPs from SxA
FileS3 is for polymorphics markers common to both populations.
FileS4 is just Chromosome 5A
FileS5 is everything except 5A

"""

import numpy as np
import os
import glob
import parse

import pandas as pd

_this_dir = os.path.dirname(__file__)

FILES = glob.glob(os.path.join(_this_dir, 'FileS*.xlsx'))

PHENO_COLS = ['id', 'TYM', 'SGNC', 'SGNO']

def main():
    # Loop over each input file.
    for filepath in FILES:
        # Figure out which file we're working with.
        _, file_name = os.path.split(filepath)
        parsed = parse.parse('FileS{}.xlsx', file_name)
        file_number = str(parsed.fixed[0])

        # Read the file.
        df = pd.read_excel(filepath)

        # Pull out the phenotype and genotype data.
        phenos = df[PHENO_COLS]
        geno_cols = np.array(list(set(df.columns) - set(PHENO_COLS)))
        geno_cols = np.sort(geno_cols)
        print('File{}'.format(file_number), len(geno_cols))
        genos = df[geno_cols] - 1 # Convert from [0,1,2] to [-1, 0, 1]
        stuff[file_number] = geno_cols

        # Create the output directory.
        output_dir = os.path.join(_this_dir, 'File{}'.format(file_number))
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        # Write out files.
        pheno_file = os.path.join(output_dir, 'phenotypes.csv')
        phenos.to_csv(pheno_file)
        geno_file = os.path.join(output_dir, 'genotypes.csv')
        genos.to_csv(geno_file, header=False, index=False, na_rep='NaN', float_format='%1.0f')
    
if __name__ == '__main__':
    main()
