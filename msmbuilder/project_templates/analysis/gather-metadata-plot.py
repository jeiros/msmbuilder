"""Plot metadata info

{{header}}
"""

# ? include "plot_header.template"
# ? from "plot_macros.template" import xdg_open with context

import matplotlib
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from glob import glob
from msmbuilder.io import load_meta, render_meta

sns.set_style('ticks')
colors = sns.color_palette()

# Load


# Histogram of trajectory lengths
def plot_lengths(ax, meta):
    lengths_ns = meta['nframes'] * (meta['step_ps'] / 1000)
    ax.hist(lengths_ns)
    ax.set_xlabel("Lengths / ns", fontsize=16)
    ax.set_ylabel("Count", fontsize=16)

    total_label = ("Total length: {us:.2f}"
                   .format(us=np.sum(lengths_ns) / 1000))
    total_label += r" / $\mathrm{\mu s}$"
    ax.annotate(total_label,
                xy=(0.55, 0.95),
                xycoords='axes fraction',
                fontsize=18,
                va='top',
                )


# Pie graph
def plot_pie(ax, meta):
    lengths_ns = meta['nframes'] * (meta['step_ps'] / 1000)
    sampling = lengths_ns.groupby(level=0).sum()

    ax.pie(sampling,
           shadow=True,
           labels=sampling.index,
           colors=sns.color_palette(),
           )
    ax.axis('equal')


# Box plot
def plot_boxplot(ax, meta):
    meta2 = meta.copy()
    meta2['ns'] = meta['nframes'] * (meta['step_ps'] / 1000)
    sns.boxplot(
        x=meta2.index.names[0],
        y='ns',
        data=meta2.reset_index(),
        ax=ax,
    )

if __name__ == '__main__':
    meta = load_meta()
    # Plot hist
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_lengths(ax, meta)
    fig.tight_layout()
    fig.savefig("lengths-hist.pdf")
    #

    # Plot pie
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_pie(ax, meta)
    fig.tight_layout()
    fig.savefig("lengths-pie.pdf")
    #

    # Plot box
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_boxplot(ax, meta)
    fig.tight_layout()
    fig.savefig("lengths-boxplot.pdf")

    for f in glob('*meta.pickl'):
        meta = load_meta(f)
        name = ''.join(f.split('.')[0].split('_')[0:3])
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_lengths(ax, meta)
        fig.tight_layout()
        fig.savefig("%slengths-hist.pdf" % name)
        #

        # Plot pie
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_pie(ax, meta)
        fig.tight_layout()
        fig.savefig("%slengths-pie.pdf" % name)
        #

        # Plot box
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_boxplot(ax, meta)
        fig.tight_layout()
        fig.savefig("%slengths-boxplot.pdf" % name)
