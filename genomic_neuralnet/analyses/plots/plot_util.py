from __future__ import print_function

import os
import json

import numpy as np
import pandas as pd
import scipy.stats as sps
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.multicomp import pairwise_tukeyhsd 
from shelve import DbfilenameShelf
from contextlib import closing
from collections import defaultdict
from functools import partial
from sklearn.preprocessing import OneHotEncoder
from genomic_neuralnet.analyses.plots.get_significance_labels import get_labels

_this_dir = os.path.dirname(__file__)
data_dir = os.path.join(_this_dir, '..', 'shelves')
timing_dir = os.path.join(_this_dir, '..', 'timing_logs')

out_dir = _this_dir

# Dark style, with palette that is printer and colorblind friendly.
sns.set_style('dark')
palette = sns.cubehelix_palette(n_colors=4, rot=0.7, dark=0.20, light=0.6)
sns.set_palette(palette)

def _get_shelf_data(path):
    with closing(DbfilenameShelf(path, flag='r')) as shelf:
        return dict(shelf) 

def get_timing_data():
    records = []
    for file_name in os.listdir(timing_dir):
        if not file_name.endswith('.log'):
            continue
        species = file_name.split('_')[0]
        processor = file_name.split('_')[-1].split('.')[0]
        size = file_name.split('_')[-2]
        # Trait is anything that is not the species, trait, or size.
        trait = file_name[len(species) + 1:-1*((len(processor) + 4) + len(size) + 2)]
        full_path = os.path.join(timing_dir, file_name)
        for line in open(full_path, 'r'):
            try:
                time = json.loads(line)['seconds']
                records.append((species, trait, size, processor, time))
            except:
                continue # a non-json log line.

    df = pd.DataFrame.from_records(records)
    df.columns = ['species', 'trait', 'size', 'processor', 'time']
    return df

def get_nn_model_data():
    index_db_path = os.path.join(data_dir, 'index.shelf')
    index = _get_shelf_data(index_db_path)
    
    data = {}
    for name, path in index.iteritems():
        db_path = os.path.join(data_dir, path)
        data[name] = _get_shelf_data(db_path) 

    # Filter to just NN items.
    nn_keys = filter(lambda k: k.startswith('N'), data)
    data = {k:v for k,v in data.iteritems() if k in nn_keys}

    return data

def get_all_model_data():
    index_db_path = os.path.join(data_dir, 'index.shelf')
    index = _get_shelf_data(index_db_path)
    
    data = {}
    for name, path in index.iteritems():
        db_path = os.path.join(data_dir, path)
        data[name] = _get_shelf_data(db_path) 

    # Filter to just NN items.
    data = {k:v for k,v in data.iteritems()}

    return data

def get_significance_letters(accuracy_df, ordered_model_names):
    # Uses Holm-Bonferroni method (step-down procedure).
    # Goal: return a lookup (species, trait, model) -> Letter Code

    # First get a way to look up pairwise p-values.
    # (species, trait, (model_a, model_b)) -> p_value
    pval_lookup = defaultdict(lambda: defaultdict(lambda: {}))
    groups = accuracy_df.groupby(['species', 'trait'])
    for (species, trait), sub_df in groups:
        for index_1, series_1 in sub_df.iterrows():
            for index_2, series_2 in sub_df.iterrows():
                if index_1 == index_2:
                    continue # Don't compare with yourself.

                # Run all possible pairwise tests within the species/trait combo.
                model_1 = series_1['model']
                accuracies_1 = np.hstack(series_1['raw_results'])
                model_2 = series_2['model']
                accuracies_2 = np.hstack(series_2['raw_results'])

                # A paired T-test is appropriate here.
                t_stat, p_value = sps.ttest_rel(accuracies_1, accuracies_2)

                # This will be overwritten when model_1 and model_2 swap places
                # during iteration. The p-value is the same either way, 
                # so it doesn't matter if it gets overwritten.
                pval_lookup[species][trait][frozenset([model_1, model_2])] = p_value

    # Now, produce a way to look up pairwise hypothesis rejection.
    # lookup[species][trait][hypothesis (model pair)] = reject or not
    hypothesis_lookup = defaultdict(lambda: defaultdict(lambda: {}))
    
    _ALPHA = 0.05
    for species, trait_dict in pval_lookup.iteritems():
        for trait, model_sets in trait_dict.iteritems():
            # Holm-Bonferroni multiple-comparison correction. 
            # See https://en.wikipedia.org/wiki/Holm-Bonferroni_method
            hypothesis_keys, p_values = map(np.array, zip(*model_sets.iteritems()))
            sort_indexes = np.argsort(p_values)
            p_values = p_values[sort_indexes]
            hypothesis_keys = hypothesis_keys[sort_indexes]
            should_reject = []
            for p_idx, p_value in enumerate(p_values):
                criteria = _ALPHA / (len(p_values) + 1 - (p_idx + 1))
                reject = p_value > criteria 
                should_reject.append(reject)
            minimal = np.argmax(should_reject)
            rejections = np.zeros(len(p_values)).astype(bool)
            rejections[:minimal] = True
            for hypothesis, rejection in zip(hypothesis_keys, rejections):
                hypothesis_lookup[species][trait][hypothesis] = rejection

    # Assign numbers to each hypothesis 'cluster'.
    # lookup[species][trait][model] = number
    significance_number_lookup = defaultdict(lambda: defaultdict(dict))
    for species, trait_dict in hypothesis_lookup.iteritems():
        for trait, hypothesis_sets in trait_dict.iteritems(): 
            these_sig_numbers = get_labels(*zip(*hypothesis_sets.iteritems()))
            for model, sig_numbers in these_sig_numbers.iteritems():
                significance_number_lookup[species][trait][model] = sig_numbers 

    # Convert numbers to letters by sorting.
    significance_letter_lookup = defaultdict(lambda: defaultdict(lambda: {}))
    for species, trait_dict in significance_number_lookup.iteritems():
        for trait, model_dict in trait_dict.iteritems():
            class Container(object):
                pass
            obj = Container()
            obj.current_letter = 'A'
            def get_next_letter(obj):
                to_return = obj.current_letter
                obj.current_letter = chr(ord(obj.current_letter) + 1)
                return to_return
            get_next_letter = partial(get_next_letter, obj)

            number_to_letter = defaultdict(get_next_letter) 
            for model in ordered_model_names:
                model_significance_numbers = model_dict[model]
                significance_letters = map(number_to_letter.__getitem__, model_significance_numbers)
                significance_letter_lookup[model][species][trait] = ''.join(significance_letters)

    return significance_letter_lookup


def main():
    #get_timing_data() 
    pass

if __name__ == '__main__':
    main()
