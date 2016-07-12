from __future__ import print_function

from genomic_neuralnet.util import get_is_on_gpu 

class OptimizationResult(object):
    def __init__(self, df, time, species, trait, folds, runs):
        self.df = df
        self.on_gpu = get_is_on_gpu()
        self.species = species
        self.trait = trait
        self.fitting_time = time
        self.folds = folds 
        self.runs = runs

