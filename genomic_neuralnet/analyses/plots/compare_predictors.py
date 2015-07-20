import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METHOD_IDX = 0
ACCURACY_IDX = 1
STD_DEV_IDX = 2

BAR_WIDTH=0.3

this_dir = os.path.dirname(__file__)
comparison_csv_path = os.path.join(this_dir, 'compare_predictors.csv')
df = pd.DataFrame.from_csv(comparison_csv_path, index_col=None)
df['method'] = map(lambda x: x.replace('_', '\n'), df['method'])
print(df)
df = df.sort(columns='mean')
df = df.reset_index(drop=True)
print(df)

def main():
    ind = np.arange(df.shape[0]) 
    fig, ax = plt.subplots()
    ax.bar(ind, df['mean'], BAR_WIDTH, color='r', yerr=df['std_dev'] / np.sqrt(100))
    ax.set_ylim((0.0, 0.5))
    ax.set_ylabel('Accuracy (correlation with truth)') 
    ax.set_xticks(ind + BAR_WIDTH)
    ax.set_xticklabels( df['method'])
    ax.axhline(y=0.85, c='k')
    plt.show()

if __name__ == '__main__':
    main()
