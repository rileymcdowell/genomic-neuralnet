from __future__ import print_function

import os

import numpy as np
import pandas as pd
import scipy.stats as sps
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from statsmodels.stats.multicomp import pairwise_tukeyhsd 
from shelve import DbfilenameShelf
from contextlib import closing
from collections import defaultdict
from functools import partial
from sklearn.preprocessing import OneHotEncoder
from genomic_neuralnet.analyses.plots \
        import get_nn_model_data, palette, png_dir \
             , get_significance_letters

sns.set_style('darkgrid')
sns.set_context('paper')
sns.set_palette(palette)

def two_line_label((species, trait)):
    trait_name = trait.replace('_', ' ').title()
    species_name = species.title()
    return '{}\n{}'.format(species_name, trait_name)

def make_best_by_depth_dataframe(shelf_data):
    dfs = []

    num_models = len(shelf_data) 

    # Pull data from shelves.
    for model_name, optimization in shelf_data.iteritems():
        for species_trait, opt_result in optimization.iteritems():
            species, trait, gpu = tuple(species_trait.split('|'))
            df = opt_result.df
            df['depth'] = map(len, df['hidden'])
            df['species'] = species
            df['trait'] = trait
            df['gpu'] = gpu
            df['model'] = model_name.upper()
            raw_res = np.array(df['raw_results'].tolist())
            no_nan_means = np.apply_along_axis(np.nanmean, axis=1, arr=raw_res)
            df['nn_mean'] = no_nan_means
            dfs.append(df)
    
    accuracy_df = pd.concat(dfs).reset_index(drop=True)

    return accuracy_df


def make_plot(accuracy_df):

    accuracy_df = accuracy_df.sort_values(by=['species', 'trait'], ascending=[0, 1])

    species_list = accuracy_df['species'].unique()
    depths = accuracy_df['depth'].unique()
    models = ['N', 'NDO', 'NWD', 'NWDDO']

    accuracy_df['trait_id'] = -1
    trait_by_species = defaultdict(lambda: list(['', ''])) 
    for species_idx, species in enumerate(species_list):
        is_species = accuracy_df['species'] == species
        traits = accuracy_df[is_species]['trait'].unique()
        for trait_idx, trait in enumerate(traits):
            is_trait = accuracy_df['trait'] == trait
            matches = is_species & is_trait
            accuracy_df.loc[matches, 'trait_id'] = trait_idx
            trait_by_species[species][trait_idx] = trait

    violin_params = { 'palette': palette
                     , 'width': 0.8 # Almost use full width for violins.
                     , 'inner': None # Don't make boxplots inside violins.
                     , 'cut': 0 # Don't extend PDF past extremes.
                     , 'scale': 'width' 
                     , 'hue_order': models
		     , 'linewidth': 0.0 # No lines around violins.
                     , 'saturation': 1.0
                     }
    subplot_columns = ['depth', 'mean', 'model']
    g = sns.FacetGrid(accuracy_df, col="trait_id", row="species", ylim=(-0.5, 1))
    g = g.map(sns.violinplot, *subplot_columns, **violin_params) \
        .set_axis_labels("Number of Hidden Layers", "Average Accuracy")

    legend_data = g.axes[2][1].get_legend_handles_labels()
    g.axes[2][1].legend(*legend_data, loc='lower left')

    for species_idx, trait_idx in np.ndindex(g.axes.shape):
        ax = g.axes[species_idx, trait_idx]
        species = species_list[species_idx]
        trait = trait_by_species[species][trait_idx]
        ax.set_title(two_line_label((species, trait)))
        ax.get_xaxis().set_minor_locator(ticker.AutoMinorLocator(n=2))
        ax.grid(b=True, which='minor', color='w', linewidth=1.0)
        is_species = accuracy_df['species'] == species
        is_trait = accuracy_df['trait'] == trait
        for depth in depths:
            is_depth = accuracy_df['depth'] == depth
            for model in models:
                is_model = accuracy_df['model'] == model
                sub_df = accuracy_df[is_species & is_trait & is_depth & is_model]
                max_idx = sub_df['mean'].idxmax()
                best = sub_df.loc[max_idx]
                hidden = best['hidden']

                x_1 = best['depth'] - 1 # Group
                x_2 = (models.index(model) - 1.5) * violin_params['width'] * 0.25 # Model
                x_3 = -0.02 # Offset
                x = x_1 + x_2 + x_3
                y = best['mean'] + 0.05
                s = '-'.join(map(str, best['hidden']))
                text_params = { 'rotation': 45, 'ha': 'left', 'va': 'bottom' }
                ax.text(x, y, s, **text_params)

    plt.tight_layout()
    fig_path = os.path.join(png_dir, 'compare_depths.png') 
    plt.savefig(fig_path)
    plt.show()
    
def main():
    data = get_nn_model_data() 

    accuracy_df = make_best_by_depth_dataframe(data)

    make_plot(accuracy_df)

if __name__ == '__main__':
    main()
