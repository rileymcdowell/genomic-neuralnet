import os
import pandas as pd

_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

##### LOUDET ARABIDOPSIS #####

TRAIT_NAME = 'FLOSD' # Great accuracy (82-85%).
#TRAIT_NAME = 'DM3' # Poor accuracy (0.33-0.40).
_marker_path = os.path.join(_data_dir, 'loudet_arabidopsis', 'genotypes.csv')
markers = pd.DataFrame.from_csv(_marker_path, index_col=None, sep=',', header=None)
_pheno_path = os.path.join(_data_dir, 'loudet_arabidopsis', 'phenotypes.csv')
pheno = pd.DataFrame.from_csv(_pheno_path, index_col=None, sep=',')

##### CROSSA MAIZE #####

#TRAIT_NAME = 'ww_flf' # Female flowering time, well-watered environment.
#_marker_path = os.path.join(_data_dir, 'crossa_maize', 'maize', 'flowering', 'genotypes.csv')
#markers = pd.DataFrame.from_csv(_marker_path, index_col=None, sep=',', header=None)
#_pheno_path = os.path.join(_data_dir, 'crossa_maize', 'maize', 'flowering', 'phenotypes.csv')
#pheno = pd.DataFrame.from_csv(_pheno_path, index_col=None, sep=',')

#TRAIT_NAME = 'ww_yld' # Grain yield, well-watered environment.
#_marker_path = os.path.join(_data_dir, 'crossa_maize', 'maize', 'grain_yield', 'genotypes.csv')
#markers = pd.DataFrame.from_csv(_marker_path, index_col=None, sep=',', header=None)
#_pheno_path = os.path.join(_data_dir, 'crossa_maize', 'maize', 'grain_yield', 'phenotypes.csv')
#pheno = pd.DataFrame.from_csv(_pheno_path, index_col=None, sep=',')

##### RESENDE LOBLOLLY ######

#TRAIT_NAME = 'nassau_age6_CWAC' # Crown Width Across Planting age 6 high heritability .
#TRAIT_NAME = 'woodall_age4_N_lignin' # Wood Lignin age 4 low heritabilty (0.11).
#_marker_path = os.path.join(_data_dir, 'resende_loblolly', 'genotypes.csv')
#markers = pd.DataFrame.from_csv(_marker_path, index_col=None, sep=',', header=None)
#_pheno_path = os.path.join(_data_dir, 'resende_loblolly', 'phenotypes.csv')
#pheno = pd.DataFrame.from_csv(_pheno_path, index_col=None, sep=',')

##### CLEVELAND PIG #####

#TRAIT_NAME = 't1' # Very low heritability, mean=-0.045, sd=1.21, h^2=0.07
##TRAIT_NAME = 't5' # High heritability. mean=37.989, sd=60.45, h^2=0.62
#_marker_path = os.path.join(_data_dir, 'cleveland_pig', 'genotypes.csv.bz2')
#markers = pd.DataFrame.from_csv(_marker_path, index_col=None, sep=',', header=None)
#_pheno_path = os.path.join(_data_dir, 'cleveland_pig', 'phenotypes.csv')
#pheno = pd.DataFrame.from_csv(_pheno_path, index_col=None, sep=',')

##### THAVAMANIKUMAR WHEAT #####

#TRAIT_NAME = 'TYM' # Time to Young Microspore (flowering trait). 
#TRAIT_NAME = 'SGNC' # Spike Grain Number under Control conditions (yield trait). 
#_marker_path = os.path.join(_data_dir, 'thavamanikumar_wheat', 'FileS3', 'genotypes.csv')
#markers = pd.DataFrame.from_csv(_marker_path, index_col=None,  sep=',', header=None)
#_pheno_path = os.path.join(_data_dir, 'thavamanikumar_wheat', 'FileS3', 'phenotypes.csv')
#pheno = pd.DataFrame.from_csv(_pheno_path, index_col=None, sep=',')


