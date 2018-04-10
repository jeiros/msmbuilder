"""Plot the result of sampling a tICA coordinate

{{header}}
"""

# ? include "plot_header.template"
# ? from "plot_macros.template" import xdg_open with context

import matplotlib
matplotlib.use('Agg')
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from msmexplorer import plot_free_energy
from traj_utils import split_trajs_by_type
from msmbuilder.io import load_trajs, load_generic
import datetime
import os
today = datetime.date.today().isoformat()
o_dir = '{}_plots'.format(today)
if not os.path.exists(o_dir):
    os.mkdir(o_dir)


sns.set_style('ticks')
colors = sns.color_palette()

tic = 0

# Load
meta, ttrajs = load_trajs('ttrajs')
txx = np.concatenate(list(ttrajs.values()))
ttrajs_subtypes = split_trajs_by_type(ttrajs, meta)

samples_by_subtype = {
    key: load_generic('tica-dimension-{}-{}.pkl'.format(tic, ''.join(key.split())))
    for key in ttrajs_subtypes.keys()
}


def get_straj(inds):
    straj = []
    for traj_i, frame_i in inds:
        straj += [ttrajs[traj_i][frame_i, :]]
    straj = np.asarray(straj)
    return straj

straj_by_subtype = {
    key: get_straj(inds=value) for key, value in samples_by_subtype.items()
}

# Overlay sampled trajectory on histogram


def plot_sampled_traj(ax):
    plot_free_energy(txx, obs=(0, 1), vmin=1e-25, ax=ax, n_levels=6,
                     cmap='viridis', cbar=True, alpha=0.2,
                     cbar_kwargs={'format': '%d',
                                  'label': 'Free energy (kcal/mol)'})

    for k, v in straj_by_subtype.items():
        ax.scatter(v[:, 0], v[:, 1],
                   label=k)
    ax.set_xlabel("tIC 1", fontsize=16)
    ax.set_ylabel("tIC 2", fontsize=16)
    ax.legend(loc='best')


def plot_tic1_sampled_trajs(ax):
    for k, v in straj_by_subtype.items():
        ax.plot(v[:, 0], label=k)

    ax.legend()
    ax.set_xlabel('Frame ID', fontsize=16)
    ax.set_ylabel('tIC 1', fontsize=16)

# Plot overlap heatmap
fig, ax = plt.subplots(figsize=(7, 5))
plot_sampled_traj(ax)
fig.tight_layout()
fig.savefig('{}/tica-dimension-0-heatmap.pdf'.format(o_dir))
# Plot tic1 for each frame
fig, ax = plt.subplots(figsize=(7, 5))
plot_tic1_sampled_trajs(ax)
fig.tight_layout()
fig.savefig('{}/tica-dimension-0-trajs.pdf'.format(o_dir))
