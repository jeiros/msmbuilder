"""Plot sparse-tICA-results coordinates

{{header}}
"""

# ? include "plot_header.template"
# ? from "plot_macros.template" import xdg_open with context


import seaborn as sns
from matplotlib import pyplot as plt
from msmbuilder.io import load_generic, load_meta
import numpy as np
from plot_utils import figure_dims, plot_tic_loadings
import datetime
import os
import mdtraj
import pandas as pd
import matplotlib.patches as mpatches


today = datetime.date.today().isoformat()
o_dir = '{}_plots'.format(today)
if not os.path.exists(o_dir):
    os.mkdir(o_dir)


sns.set_style('ticks')
colors = sns.color_palette()

st = 10  # for smaller 2d trace plots


if __name__ == '__main__':
    # Load
    meta = load_meta()
    feat = load_generic('feat.pkl')
    sptica_list = load_generic('sptica_list.pkl')
    title_list = [
        r'tICA $(\rho=0)$',
        r'Sparse tICA $(\rho=10^{-4})$',
        r'Sparse tICA $(\rho=10^{-3})$',
        r'Sparse tICA $(\rho=10^{-2})$'
    ]

    tic1_patch = mpatches.Patch(color='tab:blue', label='tIC1')
    tic2_patch = mpatches.Patch(color='tab:orange', label='tIC2')
    tic3_patch = mpatches.Patch(color='tab:green', label='tIC3')

    f, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, sharey=True, figsize=(9.6, 2.4))
    for i, ax in enumerate([ax0, ax1, ax2, ax3]):
        plot_tic_loadings(sptica_list[i], ax, alpha=0.8)
        ax.set(title=title_list[i])
        ax.set_ylim([-1.1, 1.1])
        ax.annotate(
            xy=(0, -0.9),
            s=r'$\sum_i^5\hat\lambda_i={:.3f}$'.format(sptica_list[i].eigenvalues_.sum())
        )
    f.legend(handles=[tic1_patch, tic2_patch, tic3_patch], loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.045), frameon=False)
    f.tight_layout()
    f.savefig('{}/sparse_tica.pdf'.format(o_dir), bbox_inches='tight')

    # Reporting of top tICS
    traj = mdtraj.load(meta.iloc[0]['traj_fn'], top=meta.iloc[0]['top_fn'])
    df_feat = pd.DataFrame(feat.describe_features(traj))

    sptica = sptica_list[-1]  # Use last sparse tica from the list (strongest rho value)
    tic1_loads = list(np.nonzero(sptica.components_[0, :])[0])  # See what features are non-zero
    df_important = df_feat.iloc[tic1_loads]
    df_important.to_html('important-sparse-tICS.pandas.html')
