import os
import pandas as pd

_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

# Arabidopsis Data
#TRAIT_NAME = 'FLOSD'
#_marker_path = os.path.join(_data_dir, 'arabidopsis', 'genotypes.csv')
#markers = pd.DataFrame.from_csv(_marker_path, index_col=None, sep=',', header=None)
#_pheno_path = os.path.join(_data_dir, 'arabidopsis', 'phenotypes.csv')
#pheno = pd.DataFrame.from_csv(_pheno_path, index_col=None, sep=',')

# Crossa Maize Flowering Data
#TRAIT_NAME = 'ww_flf' # Female flowering time, well-watered environment.
#_marker_path = os.path.join(_data_dir, 'crossa', 'maize', 'flowering', 'genotypes.csv')
#markers = pd.DataFrame.from_csv(_marker_path, index_col=None, sep=',', header=None)
#_pheno_path = os.path.join(_data_dir, 'crossa', 'maize', 'flowering', 'phenotypes.csv')
#pheno = pd.DataFrame.from_csv(_pheno_path, index_col=None, sep=',')

# Crossa Maize Yield Data
TRAIT_NAME = 'ss_yld' # Grain yield, severe stress environment.
_marker_path = os.path.join(_data_dir, 'crossa', 'maize', 'grain_yield', 'genotypes.csv')
markers = pd.DataFrame.from_csv(_marker_path, index_col=None, sep=',', header=None)
_pheno_path = os.path.join(_data_dir, 'crossa', 'maize', 'grain_yield', 'phenotypes.csv')
pheno = pd.DataFrame.from_csv(_pheno_path, index_col=None, sep=',')
