from __future__ import print_function

import os

import numpy as np
import pandas as pd
import scipy.stats as sps
import matplotlib.pyplot as plt
import seaborn as sns
import re

from statsmodels.stats.multicomp import pairwise_tukeyhsd 
from shelve import DbfilenameShelf
from contextlib import closing
from collections import defaultdict
from functools import partial
from sklearn.preprocessing import OneHotEncoder
from genomic_neuralnet.analyses.plots \
        import get_all_model_data, palette, out_dir \

sns.set_style('dark')
sns.set_palette(palette)

def _cleanup(name):
    return name.replace('_', ' ').title()

def make_model_dataframe(shelf_data):
    records = []

    num_models = len(shelf_data) 

    for model_name, optimization in shelf_data.iteritems():
        for species_trait, opt_result in optimization.iteritems():
            species, trait, proc = tuple(species_trait.split('|'))
            if proc == 'gpu':
                pass # Skip GPU trained datasets, they're duplicates of the CPU sets.
            max_fit_index = opt_result.df['mean'].idxmax()
            best_fit = opt_result.df.loc[max_fit_index]
            mean_acc = best_fit.loc['mean']
            count = opt_result.folds * opt_result.runs
            raw_results = best_fit.loc['raw_results']
            assert len(raw_results) == count # Double-check this.
            records.append((species, trait, model_name, mean_acc))
    
    model_df = pd.DataFrame.from_records(records)
    model_df.columns = ['Species', 'Trait', 'Model', 'Accuracy']

    for column in ('Species', 'Trait'):
        model_df[column] = model_df[column].apply(_cleanup)

    return model_df

def max_bold_format(max_vals, num):
    if num in max_vals:
        return r"\textbf{}{:0.2f}{}".format('{', num, '}')
    else:
        return '{:0.2f}'.format(num)

LEVEL_ORDER = [ 'OLS', 'RR', 'LASSO' #, 'RBF'
              , 'EN', 'BRR', 'N', 'NWD'
              , 'NDO', 'NWDDO'
              ]

LEVEL_SIZES = { 1:'m{1.6em}'
              , 2:'m{1.6em}'
              , 3:'m{1.6em}'
              , 4:'m{2.2em}'
              , 5:'m{2.2em}'
              }


def write_latex(df):
    
    n_cols = len(df.columns)

    lines = []
    column_format = ' '.join(map(LEVEL_SIZES.get, map(len, LEVEL_ORDER)))
    lines.append('\\begin{{tabularx}}{{\\textwidth}}{{ m{{4.8em}} m{{4.8em}} {} }}'.format(column_format))

    # Write multi-level headers.
    lines.append('\\hline')
    header1 = '\\multicolumn{{{}}}{{c}}{{Accuracy}}'.format(n_cols)
    lines.append('\\header & & {} \\\\'.format(header1))
    columns = set(df.columns.levels[1])
    columns = list(columns.intersection(set(LEVEL_ORDER)))
    columns.sort(key=lambda x: LEVEL_ORDER.index(x))
    header2 = ' & '.join(columns)
    lines.append('\\header & & {} \\\\'.format(header2))
    lines.append('\\hline')
    header_3 = ' '.join(('&',)*n_cols)
    lines.append('\\header Species & Trait {} \\\\'.format(header_3))
    lines.append('\\hline')
    # Write table data. 
    idx = 0
    for (species, trait), row in df.iterrows():
        row = row.reset_index(drop=True, level=0)
        row = row.loc[LEVEL_ORDER]
        row_data = []
        if idx % 2 == 0:
            row_data.append(species)
        else:    
            row_data.append(' ')
        row_data.append(trait)

        max_model = row.max()
        for val in row.values:
            if val == max_model:
                row_data.append('\\underline{{{:0.2f}}}'.format(val))
            else:
                row_data.append('{:0.2f}'.format(val))


        lines.append(' & '.join(row_data) + ' \\\\')
        if idx % 2 == 1:
            lines.append('\\hline')
        idx += 1

    # Write end of table.
    lines.append('\\end{tabularx}')

    # Write table to file.
    fig_path = os.path.join(out_dir, 'model_comparison.tex')
    with open(fig_path, 'w') as f:
        f.write('\n'.join(lines))

def make_table(model_df):
    # TODO: Apply format specs to the dataframe.
    
    model_df = model_df.set_index(keys=['Species', 'Trait', 'Model'])
    model_df = model_df.unstack()

    return model_df
    
def main():
    data = get_all_model_data() 
    model_df = make_model_dataframe(data)
    model_df = make_table(model_df)
    write_latex(model_df)

if __name__ == '__main__':
    main()
