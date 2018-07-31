"""Plot populations and eigvectors from microstate MSM

{{header}}
Meta
----
depends:
 - kmeans.pickl
 - ../ttrajs
"""

# ? include "plot_header.template"
# ? from "plot_macros.template" import xdg_open with context

import datetime
import os

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from plot_utils import plot_ergodic_subspace, plot_cluster_centers
from msmadapter.plot_utils import plot_tica_landscape
from msmbuilder.io import load_trajs, load_generic
from msmexplorer.utils import msme_colors
from msmexplorer.palettes import msme_rgb
from matplotlib import pyplot as pp

today = datetime.date.today().isoformat()
o_dir = '{}_plots'.format(today)
if not os.path.exists(o_dir):
    os.mkdir(o_dir)

sns.set_style('ticks')
colors = sns.color_palette()


def find_unique_clusters(cluster_dict, system):
    """
    Finds the cluster IDs which are *unique* to a given system.

    Parameters
    ----------
    cluster_dict: dict of np.arrays of ints, The cluster IDs for each system
    system: str, Must be a key of cluster_dict: this is the system we will fin
    the unique cluster IDs for
    Returns
    -------
    unique: np.array(dtype=int), The cluster IDs which have been found to be
        unique for system

    Example
    -------
    foo = {
        'A':np.array([1,2,3,7]),
        'B':np.array([6,7,3,8]),
        'C':np.array([2,3,5,8,13]),
        'D':np.array([13,14,16,1,2])
    }
    find_unique_clusters(foo, 'A')
    >>> array([], dtype=int64)
    find_unique_clusters(foo, 'D')
    >>> array([14, 16])
    """
    union = np.array([], dtype=int)
    for key in cluster_dict.keys():
        if system != key:
            union = np.union1d(union, cluster_dict[key])
    unique = np.setdiff1d(cluster_dict[system], union)
    return unique


@msme_colors
def plot_timescales(msm, n_timescales=None, error=None, sigma=2,
                    color_palette=None, xlabel=None, ylabel=None, ax=None,
                    dt_traj_ns=None):
    """
    Plot MSM timescales spectral diagram.
    Parameters
    ----------
    msm : msmbuilder.msm
        MSMBuilder MarkovStateModel
    n_timescales : int, optional
        Number of timescales to plot
    error : array-like (float), optional
        associated errors for each timescales
    sigma : float, optional
        significance level for default error bars
    color_palette: list or dict, optional
        Color palette to apply
    xlabel : str, optional
        x-axis label
    ylabel : str, optional
        y-axis label
    ax : matplotlib axis, optional (default: None)
        Axis to plot on, otherwise uses current axis.
    dt_traj_ns: float, Timestep of trajs in ns (optional)
    Returns
    -------
    ax : matplotlib axis
        matplotlib figure axis
    """

    if dt_traj_ns is None:
        dt = 1
    else:
        dt = dt_traj_ns

    if hasattr(msm, 'all_timescales_'):
        timescales = msm.all_timescales_.mean(0) * dt
        if not error:
            error = (msm.all_timescales_.std(0) * dt /
                     msm.all_timescales_.shape[0] ** 0.5 * dt)
    elif hasattr(msm, 'timescales_'):
        timescales = msm.timescales_ * dt
        if not error:
            error = np.nan_to_num(msm.uncertainty_timescales() * dt)

    if n_timescales:
        timescales = timescales[:n_timescales]
        error = error[:n_timescales]
    else:
        n_timescales = timescales.shape[0]

    ymin = 10 ** np.floor(np.log10(np.nanmin(timescales)))
    ymax = 10 ** np.ceil(np.log10(np.nanmax(timescales)))

    if not ax:
        f, ax = pp.subplots(1, 1, figsize=(2, 4))
    if not color_palette:
        color_palette = list(msme_rgb.values())

    for i, item in enumerate(zip(timescales, error)):
        t, s = item
        color = color_palette[i % len(color_palette)]
        ax.errorbar([0, 1], [t, t], c=color)
        if s:
            for j in range(1, sigma + 1):
                ax.fill_between([0, 1], y1=[t - j * s, t - j * s],
                                y2=[t + j * s, t + j * s],
                                color=color, alpha=0.2 / j)

    ax.xaxis.set_ticks([])
    if xlabel:
        ax.xaxis.set_label_text(xlabel, size=14)
        ax.xaxis.labelpad = 14
    if ylabel:
        ax.yaxis.set_label_text(ylabel, size=14)
    ax.set_yscale('log')
    ax.set_ylim([ymin, ymax])

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    return f, ax


def plot_microstates(ax, msm, obs=(0, 1), eigenvector=1,
                     clabel='First Dynamical Eigenvector'):
    ax.hexbin(txx[:, 0], txx[:, 1],
              cmap='Greys',
              mincnt=1,
              bins='log',
              )

    scale = 100 / np.max(msm.populations_)
    add_a_bit = 5
    prune = clusterer.cluster_centers_[:, obs]
    c = ax.scatter(prune[msm.state_labels_, 0],
                   prune[msm.state_labels_, 1],
                   s=scale * msm.populations_ + add_a_bit,
                   c=msm.left_eigenvectors_[:, eigenvector],
                   cmap='RdBu'
                   )
    ax.set_xlabel("tIC 1", fontsize=16)
    ax.set_ylabel("tIC 2", fontsize=16)
    plt.colorbar(c, label=clabel)
    return ax


if __name__ == '__main__':
    #  Load
    meta, ttrajs = load_trajs('ttrajs')
    timestep = int(meta['step_ps'].unique()) / 1e6  # from ps to us
    clusterer = load_generic('clusterer.pkl')
    txx = np.concatenate(list(ttrajs.values()))
    msms_type = load_generic('msm_dict.pkl')

    # Build a dictionary with the cluster IDs in the ergodic space for each system
    clusters_by_type = {}
    for name, msm in msms_type.items():
        clusters_by_type[name] = np.array(list(msm.mapping_.keys()))

    i = 0
    for name, msm in msms_type.items():
        # Ergodic plots
        fig, axergo = plt.subplots(figsize=(7, 5))
        plot_tica_landscape(ttrajs, ax=axergo, alpha=.4)
        axergo = plot_ergodic_subspace(msm, clusterer, color=colors[i], alpha=1, ax=axergo)
        i += 1
        axergo.set_title(name)
        fig.tight_layout()
        fig.savefig('{}'.format(o_dir) + '/ergodic-space' + '_'.join(name.split()) + '.pdf')
        fig.clf()

        # Transition plots with source and sinks
        for ev, ev_name in zip(range(1, 4), ['1st', '2nd', '3rd']):
            print(name, ev, ev_name)
            f, ax = plt.subplots(figsize=(7, 5))
            ax.set_title(name)
            ax = plot_microstates(ax, msm=msm, eigenvector=ev, clabel='{} dynamical eigenvector'.format(ev_name))
            f.tight_layout()
            f.savefig('{}/{}'.format(o_dir, ev_name) + '_'.join(name.split()) + '.pdf')
            f.clf()

        # Plot timescales of MSM
        f, ax = plot_timescales(msm, dt_traj_ns=timestep, ylabel=r'Timescale ($\mu s$)')
        f.tight_layout()
        ax.set_title(name)
        f.savefig('{}'.format(o_dir) + '/timescales' + '_'.join(name.split()) + '.pdf')
        f.clf()

        # Plot clusters which are only sampled for this system
        unique_cluster_IDs = find_unique_clusters(clusters_by_type, name)
        f, ax = plt.subplots(figsize=(7, 5))
        ax = plot_cluster_centers(clusterer, unique_cluster_IDs, txx, ax=ax)
        ax.set_title(name)
        f.tight_layout()
        f.savefig('{}'.format(o_dir) + '/unique-clusters' + '_'.join(name.split()) + '.pdf')
