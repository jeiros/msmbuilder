"""Plot sparse-tICA-results coordinates

{{header}}
"""

# ? include "plot_header.template"
# ? from "plot_macros.template" import xdg_open with context



import seaborn as sns
from matplotlib import pyplot as plt
from msmbuilder.io import load_generic

from plot_utils import figure_dims, plot_tic_loadings
import datetime
import os


today = datetime.date.today().isoformat()
o_dir = '{}_plots'.format(today)
if not os.path.exists(o_dir):
    os.mkdir(o_dir)


sns.set_style('ticks')
colors = sns.color_palette()

st = 10  # for smaller 2d trace plots


if __name__ == '__main__':
    # Load
    sptica_list = load_generic('sptica_list.pkl')
    title_list = [
        r'tICA $(\rho=0)$',
        r'Sparse tICA $(\rho=10^{-4})$',
        r'Sparse tICA $(\rho=10^{-3})$',
        r'Sparse tICA $(\rho=10^{-2})$'
    ]
    f, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, sharey=True, figsize=(9.6, 2.4))
    for i, ax in enumerate([ax0, ax1, ax2, ax3]):
        plot_tic_loadings(sptica_list[i], ax, alpha=0.8)
        ax.set(title=title_list[i])
        ax.set_ylim([-1.1, 1.1])
        ax.annotate(
            xy=(0, -0.9),
            s=r'$\sum_i^5\hat\lambda_i={:.3f}$'.format(sptica_list[i].eigenvalues_.sum())
        )
    f.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.045), frameon=False)
    f.tight_layout()
    fig.savefig('{}/sparse_tica.pdf'.format(o_dir))
