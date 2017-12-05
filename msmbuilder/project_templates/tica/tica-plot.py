"""Plot tICA-transformed coordinates

{{header}}
"""

# ? include "plot_header.template"
# ? from "plot_macros.template" import xdg_open with context


import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from msmbuilder.io import load_trajs, load_generic
from msmexplorer import plot_free_energy, plot_trace2d
from plot_utils import plot_tica_timescales, plot_singletic_trajs, plot_overlayed_types
from traj_utils import split_trajs_by_type
from plot_utils import figure_dims
sns.set_style('ticks')
colors = sns.color_palette()

st = 10  # for smalled 2d trace plots


def bar_plot_tic_loadings(tica, ax=None, n_tics=3):
    if ax is None:
        ax = plt.gca()

    n_feats_plot = tica.n_features + 1
    for i in range(n_tics):
        ax.bar(
            range(1, n_feats_plot),
            tica.components_[i, :],
            alpha=0.5,
            label='tIC{}'.format(i + 1)
        )
    xx = np.arange(1, n_feats_plot)
    ax.set_xticks(xx)
    ax.set_xlim((0, n_feats_plot))
    ax.legend(loc='best')
    return ax


if __name__ == '__main__':
    # Load
    tica = load_generic('tica.pkl')
    meta, ttrajs = load_trajs('ttrajs')
    txx = np.concatenate(list(ttrajs.values()))
    ttrajs_subtypes = split_trajs_by_type(ttrajs, meta)
    timestep = meta['step_ps'].unique()[0] / 1000

    # Plot 1 (tICA timescales)
    fig, ax = plt.subplots(figsize=(3, 5))
    plot_tica_timescales(tica=tica, meta=meta, ax=ax, color='tarragon')
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig('tica_timescales.pdf')

    # Plot 2 (tICA landscape)
    fig, ax = plt.subplots(figsize=figure_dims(600, factor=0.9))
    plot_free_energy(txx, obs=(0, 1), n_levels=6, vmin=1e-25, ax=ax,
                     cmap='viridis', cbar=True, xlabel='tIC 1', ylabel='tIC 2',
                     cbar_kwargs={'format': '%d', 'label': 'Free energy (kcal/mol)'})
    fig.tight_layout()
    fig.savefig('tica_landscape.pdf')

    # Plot 3 (Three first tIC trajs for each system type)
    for system_type in ttrajs_subtypes.keys():
        plot_singletic_trajs(ttrajs, meta, system_type, stride=st, title=system_type, alpha=0.5,
                             xlabel='Time (ns)', figsize=figure_dims(600, factor=0.9))
        f = plt.gcf()
        f.tight_layout()
        f.savefig('tica_indiv_{}.pdf'.format(''.join(system_type.split())))

    # Plot 4 (Each type overlayed on landscape)
    f, ax = plt.subplots(figsize=figure_dims(600, factor=0.9))
    _ = plot_overlayed_types(
        ttrajs=ttrajs, meta=meta,
        ax=ax, xlabel='tIC 1', ylabel='tIC 2',
        plot_free_energy_kwargs={
            'alpha': .5,
            'n_levels': 5,
            'cbar': True,
            'cmap': 'viridis',
            'vmax': 5,
            'cbar_kwargs': {'format': '%d', 'label': 'Free energy (kcal/mol)'}
        }
    )
    f.tight_layout()
    f.savefig("tica_landscape_traj_types_overlayed.pdf")

    # Plots 5-7 (2D traces of trajs inside each type)
    for k, v in ttrajs_subtypes.items():
        f, ax = plt.subplots(figsize=figure_dims(600, factor=0.9))
        plot_free_energy(txx, obs=(0, 1), n_levels=6, vmin=1e-25, ax=ax,
                         cmap='viridis', cbar=True, xlabel='tIC 1', ylabel='tIC 2',
                         cbar_kwargs={'format': '%d', 'label': 'Free energy (kcal/mol)'})
        plot_trace2d(
            data=[val[::st] for val in v.values()],
            ts=timestep * st, ax=ax
        )
        ax.set_title(k)
        f.tight_layout()
        f.savefig('tica_2dtrace_{}.pdf'.format(''.join(k.split())))

    # Plot 8 (tICA loadings)
    f, ax = plt.subplots(figsize=(7, 5))
    ax = bar_plot_tic_loadings(tica=tica, ax=ax)
    ax.set(ylabel='Component')
    f.tight_layout()
    f.savefig('tica_loadings.pdf')
