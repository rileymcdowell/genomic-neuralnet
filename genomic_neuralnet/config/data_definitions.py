from __future__ import print_function
import os
import pandas as pd

_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

class DataDefinition(object):
    def __init__(self, trait_name, *args, **kwargs):
        """ 
        Defines a datatype to predict.

        Parameters:

            trait_name: The name of the trait column in phenotypes.csv to predict.

            *args:      The tokens of the path to the data directory that contains 
                        phenotypes.csv and genotypes.csv.

            **kwargs:   Can contain compressed_geno=True to load a bz2 compressed 
                        genotypes file.
        """
        self.trait_name = trait_name
        self.marker_path = os.path.join(*((_data_dir,) + args + ('genotypes.csv',)))
        # Support .bz2 extension on genotype file.
        if 'compressed_geno' in kwargs and kwargs['compressed_geno'] == True:
            self.marker_path = '.'.join([self.marker_path, 'bz2'])
        self.pheno_path = os.path.join(*((_data_dir,) + args + ('phenotypes.csv',)))

        self._markers = None
        self._pheno = None

    @property
    def markers(self):
        if self._markers is None:
            params = {'index_col': None, 'sep': ',', 'header': None}
            self._markers = pd.DataFrame.from_csv(self.marker_path, **params)
        return self._markers

    @property
    def pheno(self):
        if self._pheno is None:
            params = {'index_col': None, 'sep': ','}
            self._pheno = pd.DataFrame.from_csv(self.pheno_path, **params)[self.trait_name]
        return self._pheno

# Populate the data into a data dictionary.
data = { 'arabidopsis': {}
       , 'maize': {}
       , 'pig': {}
       , 'loblolly': {}
       , 'wheat': {}
       } 

##### LOUDET ARABIDOPSIS #####
# Short day flowering - Great accuracy (82-85%).
data['arabidopsis']['flowering'] = DataDefinition('FLOSD', 'loudet_arabidopsis')
# Dry matter accumulation - Poor accuracy (33%-40%).
data['arabidopsis']['dry_matter'] = DataDefinition('DM3', 'loudet_arabidopsis')

##### CROSSA MAIZE #####
# Female flowering time, well-watered environment.
data['maize']['flowering'] = DataDefinition('ww_flf', 'crossa_maize', 'maize', 'flowering')
# Grain yield, well-watered environment.
data['maize']['grain_yield'] = DataDefinition('ww_yld', 'crossa_maize', 'maize', 'grain_yield')

##### RESENDE LOBLOLLY ######
# Crown Width Across Planting age 6 high heritability.
data['loblolly']['crown_width'] = DataDefinition('nassau_age6_CWAC', 'resende_loblolly')
# Wood lignin age 4 low heritability.
data['loblolly']['lignin'] = DataDefinition('woodall_age4_N_lignin', 'resende_loblolly')

##### CLEVELAND PIG #####
# Very low heritability, mean=-0.045, sd=1.21, h^2=0.07
data['pig']['trait_1'] = DataDefinition('t1', 'cleveland_pig', compressed_geno=True)
#TRAIT_NAME = 't5' # High heritability. mean=37.989, sd=60.45, h^2=0.62
data['pig']['trait_5'] = DataDefinition('t5', 'cleveland_pig', compressed_geno=True)

##### THAVAMANIKUMAR WHEAT #####
#TRAIT_NAME = 'TYM' # Time to Young Microspore (flowering trait). 
data['wheat']['time_young_microspore'] = DataDefinition('TYM', 'thavamanikumar_wheat', 'FileS3') 
#TRAIT_NAME = 'SGNC' # Spike Grain Number under Control conditions (yield trait). 
data['wheat']['spike_grain_number'] = DataDefinition('SGNC', 'thavamanikumar_wheat', 'FileS3') 


