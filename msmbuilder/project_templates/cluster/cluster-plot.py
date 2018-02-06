"""Plot cluster centers on tICA coordinates

{{header}}
"""

# ? include "plot_header.template"
# ? from "plot_macros.template" import xdg_open with context

import matplotlib
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from msmbuilder.io import load_trajs, load_generic

import datetime
import os
today = datetime.date.today().isoformat()
o_dir = '{}_plots'.format(today)
if not os.path.exists(o_dir):
    os.mkdir(o_dir)

sns.set_style('ticks')
colors = sns.color_palette()

# Load
clusterer = load_generic('clusterer.pkl')

meta, ttrajs = load_trajs('ttrajs')
txx = np.concatenate(list(ttrajs.values()))


def plot_cluster_centers(ax):
    ax.hexbin(txx[:, 0], txx[:, 1],
              cmap=sns.cubehelix_palette(as_cmap=True),
              mincnt=1,
              bins='log',
              )
    ax.scatter(clusterer.cluster_centers_[:, 0],
               clusterer.cluster_centers_[:, 1],
               s=40, c=colors[0],
               )
    ax.set_xlabel("tIC 1", fontsize=16)
    ax.set_ylabel("tIC 2", fontsize=16)


# Plot 1
fig, ax = plt.subplots(figsize=(7, 5))
plot_cluster_centers(ax)
fig.tight_layout()
fig.savefig('{}/clusterer-centers.pdf'.format(o_dir))
#

# {{xdg_open('clusterer-centers.pdf')}}
