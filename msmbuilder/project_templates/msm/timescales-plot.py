"""Plot implied timescales vs lagtime

{{header}}
"""

# ? include "plot_header.template"
# ? from "plot_macros.template" import xdg_open with context

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from msmbuilder.io import load_generic, load_meta
from plot_utils import figure_dims
sns.set_style('ticks')
colors = sns.color_palette()

# Implied timescales vs lagtime


def plot_timescales(ax, timescales, ylabel=True):
    n_timescales = int((len(timescales.columns) - 2) / 2)
    for i in range(n_timescales):
        ax.errorbar(
            x=timescales['lag_time'],
            y=timescales['timescale_{}'.format(i)] * 1000,  # in ns
            yerr=timescales['timescale_{}_unc'.format(i)],
            label=None,  # pandas be interfering
            fmt='o',
        )

    xmin, xmax = ax.get_xlim()

    xx = np.linspace(xmin, xmax)
    ax.plot(xx, xx, color=colors[2], label='$y=x$')
    ax.legend(loc='best', fontsize=14)
    ax.set_xlabel('Lag Time (ns)', fontsize=18)
    if ylabel:
        ax.set_ylabel('Implied Timescales (ns)', fontsize=18)
    ax.set_yscale('log')
    ax.set_xscale('log')


def plot_trimmed(ax, timescales):
    ax.plot(timescales['lag_time'],
            timescales['percent_retained'],
            'o-',
            label=None,  # pandas be interfering
            )
    ax.axhline(100, color='k', ls='--', label='100%')
    ax.legend(loc='best', fontsize=14)
    ax.set_xlabel('Lag Time (ns)', fontsize=18)
    ax.set_ylabel('Retained (%)', fontsize=18)
    ax.set_ylim((0, 105))
    ax.set_xscale('log')


# Load
meta = load_meta()
for system in meta.type.unique():
    system_name = ''.join(system.split())
    df = load_generic('{}timescales.pandas.pkl'.format(system_name))

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=figure_dims(2000))
    plt.suptitle(system)
    plot_trimmed(ax1, df)
    plot_timescales(ax2, df)
    f.savefig('{}timescales.pdf'.format(system_name))
