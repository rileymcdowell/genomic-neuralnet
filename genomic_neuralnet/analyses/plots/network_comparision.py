from __future__ import print_function

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from shelve import DbfilenameShelf
from contextlib import closing
from collections import defaultdict
from functools import partial

_this_dir = os.path.dirname(__file__)
data_dir = os.path.join(_this_dir, '..', 'shelves')

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
    return '{}\n{}'.format(species_name, trait_name)

def make_plot(accuracy_df):

    accuracy_df = accuracy_df.sort(['species', 'trait'], ascending=[1,0])

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

        offset = width * idx
        color = palette[idx]
        b = ax.bar(x + offset, means, width, color=color)
        e = ax.errorbar(x + offset + width/2, means, yerr=std_devs, ecolor='black', fmt='none')
        bar_sets.append(b)
        error_offsets.append(std_devs)

    # TODO: Tukey's HSD
    # https://en.wikipedia.org/wiki/Tukey%27s_range_test
    # Annotate each quad of columns.

    def label(idx, rects):
        errors = error_offsets[idx]
        for error, rect in zip(errors, rects):
            height = rect.get_height()
            ax.text( rect.get_x() + rect.get_width()/2.
                   , height + error + 0.02
                   , 'A'
                   , ha='center'
                   , va='bottom')

    [label(idx, b) for idx, b in enumerate(bar_sets)]

    # Axis labels (layer 1).
    ax.set_ylabel('Accuracy')
    ax.set_xticks(x + width / 2 * len(models))
    ax.set_xticklabels(map(string_to_label, zip(species_list, trait_list)))
    ax.set_xlim((0 - width / 2, len(trait_list)))
    ax.set_ylim((0, 1))

    # Legend
    ax.legend(bar_sets, list(models))

    #plt.savefig('network_comparison.png')
    plt.show()

def make_dataframe(shelf_data):
    data_dict = defaultdict(partial(defaultdict, dict)) 

    num_models = len(shelf_data) 

    for model_name, optimization in shelf_data.iteritems():
        for species_trait, opt_result in optimization.iteritems():
            species, trait = tuple(species_trait.split('|'))
            max_fit_index = opt_result.df['mean'].idxmax()
            best_fit = opt_result.df.loc[max_fit_index]
            mean_acc = best_fit.loc['mean']
            sd_acc = best_fit.loc['std_dev']
            hidden = best_fit.loc['hidden']
            data_dict[species][trait][model_name] = (mean_acc, sd_acc, hidden)
    
    # Add species column. Repeat once per trait per model (2*num models).
    accuracy_df = pd.DataFrame({'species': np.repeat(data_dict.keys(), num_models*2)})

    # Add trait column.
    flattened_data = []
    for species, trait_dict in data_dict.iteritems():
        for trait, model_dict in trait_dict.iteritems():
            for model, (mean, sd, hidden) in model_dict.iteritems():
                flattened_data.append((trait, model, mean, sd, hidden))
    accuracy_df['trait'], accuracy_df['model'], accuracy_df['mean'], \
        accuracy_df['sd'], accuracy_df['hidden'] = zip(*flattened_data)

    return accuracy_df
    
def main():
    data = get_nn_model_data() 

    accuracy_df = make_dataframe(data)

    make_plot(accuracy_df)

if __name__ == '__main__':
    main()
