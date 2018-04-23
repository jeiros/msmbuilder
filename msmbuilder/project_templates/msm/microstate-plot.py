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
from plot_utils import plot_ergodic_subspace

from msmbuilder.io import load_trajs, load_generic

today = datetime.date.today().isoformat()
o_dir = '{}_plots'.format(today)
if not os.path.exists(o_dir):
    os.mkdir(o_dir)

sns.set_style('ticks')
colors = sns.color_palette()

# Plot microstates


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


def plot_cluster_centers(ax):
    ax.hexbin(txx[:, 0], txx[:, 1],
              cmap='Greys',
              mincnt=1,
              bins='log',
              alpha=0.3
              )
    ax.scatter(clusterer.cluster_centers_[:, 0],
               clusterer.cluster_centers_[:, 1],
               s=2, c='black',
               )
    ax.set_xlabel("tIC 1", fontsize=16)
    ax.set_ylabel("tIC 2", fontsize=16)
    return ax


# Load
meta, ttrajs = load_trajs('ttrajs')
clusterer = load_generic('clusterer.pkl')
txx = np.concatenate(list(ttrajs.values()))
msms_type = load_generic('msm_dict.pkl')

# Ergodic plots
i = 0
for name, msm in msms_type.items():
    fig, axergo = plt.subplots(figsize=(7, 5))
    axergo.set_title('tICA ergodic subspace')
    axergo = plot_cluster_centers(axergo)
    axergo = plot_ergodic_subspace(msm, clusterer, color=colors[i], alpha=1, ax=axergo)
    i += 1
    axergo.set_title('{}'.format(name))
    fig.tight_layout()
    fig.savefig('{}'.format(o_dir) + '/ergodic-space' + '_'.join(name.split()) + '.pdf')
    fig.clf()

# Transition plots with source and sinks
for name, msm in msms_type.items():
    for ev, ev_name in zip(range(1, 4), ['1st', '2nd', '3rd']):
        print(name, ev, ev_name)
        f, ax = plt.subplots(figsize=(7, 5))
        ax.set_title(name)
        ax = plot_microstates(ax, msm=msm, eigenvector=ev, clabel='{} dynamical eigenvector'.format(ev_name))
        f.tight_layout()
        f.savefig('{}/{}'.format(o_dir, ev_name) + '_'.join(name.split()) + '.pdf')
        f.clf()
