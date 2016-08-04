from __future__ import print_function

import os

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
from get_significance_labels import get_labels

_this_dir = os.path.dirname(__file__)
data_dir = os.path.join(_this_dir, '..', '..', 'shelves')

# Dark style, with palette that is printer and colorblind friendly.
sns.set_style('dark')
palette = sns.cubehelix_palette(n_colors=4, rot=0.5, dark=0.30) 
sns.set_palette(palette)

def _get_shelf_data(path):
    with closing(DbfilenameShelf(path, flag='r')) as shelf:
        return dict(shelf) 

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

def string_to_label((species, trait)): 
    trait_name = trait.replace('_', ' ').title()
    species_name = species.title()
    if trait_name.count(' ') > 1:
        trait_name = trait_name.replace(' ', '\n')
    return '{}\n{}'.format(species_name, trait_name)

def make_dataframe(shelf_data):
    data_dict = defaultdict(partial(defaultdict, dict)) 

    num_models = len(shelf_data) 

    for model_name, optimization in shelf_data.iteritems():
        for species_trait, opt_result in optimization.iteritems():
            species, trait, gpu = tuple(species_trait.split('|'))
            max_fit_index = opt_result.df['mean'].idxmax()
            best_fit = opt_result.df.loc[max_fit_index]
            mean_acc = best_fit.loc['mean']
            sd_acc = best_fit.loc['std_dev']
            hidden = best_fit.loc['hidden']
            count = opt_result.folds * opt_result.runs
            raw_results = best_fit.loc['raw_results']
            data_dict[species][trait][model_name] = (mean_acc, sd_acc, count, raw_results, hidden)
    
    # Add species column. Repeat once per trait per model (2*num models).
    accuracy_df = pd.DataFrame({'species': np.repeat(data_dict.keys(), num_models*2)})

    # Add trait column.
    flattened_data = []
    for species, trait_dict in data_dict.iteritems():
        for trait, model_dict in trait_dict.iteritems():
            for model, (mean, sd, count, raw_res, hidden) in model_dict.iteritems():
                flattened_data.append((trait, model, mean, sd, count, raw_res, hidden))
    accuracy_df['trait'], accuracy_df['model'], accuracy_df['mean'], \
        accuracy_df['sd'], accuracy_df['count'], accuracy_df['raw_results'], \
        accuracy_df['hidden'] = zip(*flattened_data)

    return accuracy_df

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

def make_plot(accuracy_df):

    accuracy_df = accuracy_df.sort_values(by=['species', 'trait'], ascending=[1,0])

    fig, ax = plt.subplots()

    species_and_traits = accuracy_df[['species', 'trait']].drop_duplicates()
    x = np.arange(len(species_and_traits))

    models = sorted(accuracy_df['model'].unique())
    width = 0.22

    species_list = species_and_traits['species']
    trait_list = species_and_traits['trait']

    bar_sets = []
    error_offsets = []
    for idx, model in enumerate(models):
        means = accuracy_df[accuracy_df['model'] == model]['mean'].values
        std_devs = accuracy_df[accuracy_df['model'] == model]['sd'].values
        counts = accuracy_df[accuracy_df['model'] == model]['count'].values
        std_errs = std_devs / np.sqrt(counts) # SE = sigma / sqrt(N)
        # Size of 95% CI is the SE multiplied by a constant from the t distribution 
        # with n-1 degrees of freedom. [0] is the positive interval direction. 
        #confidence_interval_mult = sps.t.interval(alpha=0.95, df=counts - 1)[0]
        #confidence_interval = confidence_interval_mult * std_errs

        offset = width * idx
        color = palette[idx]
        b = ax.bar(x + offset, means, width, color=color)
        e = ax.errorbar(x + offset + width/2, means, yerr=std_errs, ecolor='black', fmt='none')
        bar_sets.append((b, model))
        error_offsets.append(std_devs)

    significance_letter_lookup = get_significance_letters(accuracy_df, ordered_model_names=models)

    def label(idx, rects, model):
        errors = error_offsets[idx]
        for error, rect, species, trait in zip(errors, rects, species_list, trait_list):
            height = rect.get_height()
            significance_letter = significance_letter_lookup[model][species][trait]
            ax.text( rect.get_x() + rect.get_width()/2.
                   , height + error + 0.02
                   , significance_letter 
                   , ha='center'
                   , va='bottom')

    [label(idx, b, m) for idx, (b,m) in enumerate(bar_sets)]

    # Axis labels (layer 1).
    ax.set_ylabel('Accuracy')
    ax.set_xticks(x + width / 2 * len(models))
    ax.set_xticklabels(map(string_to_label, zip(species_list, trait_list)))
    ax.set_xlim((0 - width / 2, len(trait_list)))
    ax.set_ylim((0, 1))

    # Legend
    print(bar_sets)
    ax.legend(map(lambda x: x[0], bar_sets), list(models))

    plt.tight_layout()
    fig_path = os.path.join(_this_dir, '..', 'network_comparison.png')
    plt.savefig(fig_path)
    plt.show()
    
def main():
    data = get_nn_model_data() 

    accuracy_df = make_dataframe(data)

    make_plot(accuracy_df)

if __name__ == '__main__':
    main()
