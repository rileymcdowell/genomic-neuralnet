from __future__ import print_function

import os

import numpy as np
import pandas as pd
import scipy.stats as sps
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as ptch
import seaborn as sns

from statsmodels.stats.multicomp import pairwise_tukeyhsd 
from shelve import DbfilenameShelf
from contextlib import closing
from collections import defaultdict
from functools import partial
from sklearn.preprocessing import OneHotEncoder
from genomic_neuralnet.analyses.plots \
        import get_timing_data, palette, out_dir \
             , get_significance_letters

from genomic_neuralnet.methods.generic_keras_net import TIMING_EPOCHS

sns.set_style('darkgrid')
sns.set_context('paper')
sns.set_palette(palette)

NUM_TRAITS = 2

def two_line_label((species, trait)):
    species_name = species.title()
    trait_name = trait.replace('_', '\n').title()
    return '{}\n{}'.format(species_name, trait_name)

def make_plot(timing_df, size, ax):

    timing_df = timing_df[timing_df['size'] == size]

    unique_species = timing_df['species'].unique()
    x = np.arange(len(unique_species))
    width = 0.2

    # Set trait ids to -1.
    timing_df['trait_id'] = -1

    # Then assign trait ids of 0 and 1 to traits so they are paired up by species.
    for species_id, spec in enumerate(unique_species):
        same_species = timing_df['species'] == spec
        species_df = timing_df[same_species]
        traits = species_df['trait'].unique()
        traits = sorted(traits)
        for idx, trait in enumerate(traits):
            same_trait = timing_df['trait'] == trait
            timing_df.loc[same_species & same_trait, 'trait_id'] = idx

    traits_and_procs = timing_df[['trait', 'trait_id', 'processor']].drop_duplicates()
    traits_and_procs = traits_and_procs.sort_values(by=['trait_id', 'processor'], ascending=[1, 1])

    trait_list = traits_and_procs['trait']
    proc_list = traits_and_procs['processor']

    bar_sets = []
    error_offsets = []
    species_labels = np.empty(len(unique_species)*NUM_TRAITS, dtype=object) 
    trait_labels = np.empty(len(unique_species)*NUM_TRAITS, dtype=object)

    data_lookup = defaultdict(lambda: defaultdict(dict))
    peak_lookup = defaultdict(lambda: defaultdict(dict))
    all_means = []

    for idx, (_, row) in enumerate(traits_and_procs[['trait_id', 'processor']].drop_duplicates().iterrows()):
        trait_id = row['trait_id']
        proc = row['processor']
        same_trait_id = timing_df['trait_id'] == trait_id
        same_proc = timing_df['processor'] == proc
        sub_df = timing_df[same_trait_id & same_proc]
        species = timing_df[same_trait_id & same_proc]['species'].unique()
        species_list = sorted(species)
        means = []
        std_errs = []

        for species_num, species in enumerate(species_list):
            same_species = timing_df['species'] == species 
            data_df = timing_df[same_species & same_trait_id & same_proc]
            trait = data_df['trait'].unique()[0]
            mean = data_df['time'].mean()
            std_dev = data_df['time'].std()
            count = len(data_df)
            std_err = std_dev / np.sqrt(count) # SE = sigma / sqrt(N)
            means.append(mean)
            std_errs.append(std_err)
            label_idx = species_num*NUM_TRAITS + trait_id 
            species_labels[label_idx] = species
            trait_labels[label_idx] = trait 
            data_lookup[species][trait][proc] = data_df['time'].values
            peak_lookup[species][trait][proc] = mean + std_err
            all_means.extend(means)

        offset = width * idx + (0 if trait_id == 0 else 0.1)
        color = palette[int(proc == 'cpu')]
        b = ax.bar(x + offset, means, width, color=color)
        e = ax.errorbar(x + offset + width/2, means, yerr=std_errs, ecolor='black', fmt='none')
        bar_sets.append((b, (trait, proc)))
        error_offsets.append(std_errs)

    label_positions = np.arange(len(unique_species) * NUM_TRAITS).astype(float) / 2 + width
    first_positions = label_positions - width/2
    second_positions = label_positions + width/2
    bar_positions = np.ravel(np.column_stack([first_positions, second_positions]))

    for x_val, species, trait in zip(label_positions, species_labels, trait_labels):
        cpu_times = data_lookup[species][trait]['cpu']
        gpu_times = data_lookup[species][trait]['gpu']
        sig_diff = sps.ttest_ind(cpu_times, gpu_times).pvalue < 0.05
        bar1_label = 'A'
        bar2_label = 'B' if sig_diff else 'A'
        ax.text( x_val - (width/2)
               , peak_lookup[species][trait]['cpu'] + 0.02
               , bar1_label
               , ha='center'
               , va='bottom')
        ax.text( x_val + (width/2)
               , peak_lookup[species][trait]['gpu'] + 0.02
               , bar2_label
               , ha='center'
               , va='bottom')

    ax.set_ylabel('Average Time (seconds)\nPer {} Epochs'.format(str(TIMING_EPOCHS/1000) + 'K'))
    ax.set_xticks(label_positions)
    ax.set_xticklabels(map(two_line_label, zip(species_labels, trait_labels))) 
    ax.set_xlim((0 - width / 2, len(x)))
    ax.set_ylim((0, np.max(all_means) * 1.15)) 
    if size == 'small':
        ax.set_title('Small Network') 
    if size == 'large':
        ax.set_title('Large Network') 

    if size == 'small':
        # Legend only on small network plot.
        cpu_patch = ptch.Patch(color=palette[int(True)], label='CPU')
        gpu_patch = ptch.Patch(color=palette[int(False)], label='GPU')
        ax.legend(handles=[cpu_patch, gpu_patch])

    return ax

SPECIES = ['arabidopsis', 'wheat', 'maize']
    
def main():
    timing_df = get_timing_data() 
    timing_df = timing_df[timing_df['species'].isin(SPECIES)]

    fig, ax_arr = plt.subplots(2, 1, sharex=True, sharey=True)
    make_plot(timing_df, size='small', ax=ax_arr[0])
    make_plot(timing_df, size='large', ax=ax_arr[1])

    # Show and save plot.
    plt.tight_layout()
    fig_path = os.path.join(out_dir, 'time_comparison.png')
    plt.savefig(fig_path, dpi=500)
    plt.show()

if __name__ == '__main__':
    main()
