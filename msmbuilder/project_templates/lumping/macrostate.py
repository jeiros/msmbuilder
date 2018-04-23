"""Make a macrostate MSM

{{header}}
"""

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from msmadapter.utils import create_folder
from traj_utils import write_cpptraj_script, split_trajs_by_type

from msmbuilder.io import load_trajs, load_generic, backup
from msmbuilder.lumping import PCCAPlus

# Settings
n_macrostates = 5
n_samples = 10
# The default colors of UCSF chimera in the order they are assigned (at least in my machine) to new models
chimera_colors = [
    '#d2b48c',  # tan
    '#87ceeb',  # sky blue
    '#dda0dd',  # plum
    '#90ee90',  # light green
    '#fa8072',  # salmon
    '#d3d3d3',  # light gray
]


def reset_keys(a_dict):
    """
    Changes a dictionary keys to be from 0 to N, where N is the number of items it has
    """
    new_dict = {}
    i = 0
    for k, v in a_dict.items():
        new_dict[i] = v
        i += 1
    return new_dict


def plot_microstate_mapping(txx, clusterer, pcca):
    # -- COLOR MESS --- #
    # In the followin code I do a dirty mapping of UCSF Chimera default colors to macrostate ID.
    # There is a faster/better/cleaner way to do this I'm sure but can't figure it out atm.
    # Also getting a discrete colorbar in matplotlib is a huge pain in the ass.
    cmap = {}
    for i, c in enumerate(chimera_colors):
        cmap[i] = c
    maping_color = []
    for i in pcca.microstate_mapping_:
        maping_color.append(cmap[i])
    # -- END OF COLOR MESS ---- #

    f, ax = plt.subplots()

    ax.hexbin(
        txx[:, 0],
        txx[:, 1],
        alpha=.3,
        mincnt=1,
        bins='log',
        cmap='Greys'
    )
    ax.scatter(
        clusterer.cluster_centers_[msm.state_labels_, 0],
        clusterer.cluster_centers_[msm.state_labels_, 1],
        s=50,
        c=maping_color,

    )
    sns.despine()
    ax.set(xlabel='tIC 1', ylabel='tIC 2')
    return f, ax


def plot_spawns_tica(txx, pcca_frames):
    # -- COLOR MESS --- #
    # In the followin code I do a dirty mapping of UCSF Chimera default colors to macrostate ID.
    # There is a faster/better/cleaner way to do this I'm sure but can't figure it out atm.
    # Also getting a discrete colorbar in matplotlib is a huge pain in the ass.
    cmap = {}
    for i, c in enumerate(chimera_colors):
        cmap[i] = c
    maping_color = []
    for i in pcca.microstate_mapping_:
        maping_color.append(cmap[i])
    # -- END OF COLOR MESS ---- #

    f, ax = plt.subplots()
    ax.hexbin(
        txx[:, 0],
        txx[:, 1],
        alpha=.3,
        mincnt=1,
        bins='log',
        cmap='Greys'
    )
    for macrostate in pcca_frames:
        ax.scatter(
            pcca_frames[macrostate][:, 0],
            pcca_frames[macrostate][:, 1],
            color=cmap[macrostate],
            marker='*',
            s=100,
            label='Macro {}'.format(macrostate)
        )
    sns.despine()
    ax.set(xlabel='tIC 1', ylabel='tIC 2')
    plt.legend(loc='best', ncol=2)
    return f, ax


# Load
meta, ttrajs = load_trajs('ttrajs')
clusterer = load_generic('clusterer.pkl')
txx = np.concatenate(list(ttrajs.values()))
msms_type = load_generic('msm_dict.pkl')
ttrajs_subtypes = split_trajs_by_type(ttrajs, meta)

print(ttrajs_subtypes.keys())

for system, msm in msms_type.items():
    system_name = ''.join(system.split())
    o_dir0 = '{}'.format(system_name)
    o_dir = '{}/{}_macrostates'.format(system_name, n_macrostates)

    backup(o_dir)

    create_folder(o_dir0)
    create_folder(o_dir)

    system_name = ''.join(system.split())
    system_meta = meta[meta.type == system]
    _, ktrajs_subtype = load_trajs('{}_ktrajs'.format(system_name), system_meta)

    ttrajs_reset = reset_keys(ttrajs_subtypes[system])
    meta_reset = system_meta.reset_index()

    pcca = PCCAPlus.from_msm(msm, n_macrostates)
    pccatrajs = pcca.transform(list(ktrajs_subtype.values()), mode='fill')

    # Plot microstate mapping to macrostate
    f, ax = plot_microstate_mapping(txx, clusterer, pcca)
    f.savefig('{}/mapping.pdf'.format(o_dir))

    # Sample states from each macrostate
    pcca_frames = {}
    for macro in range(pcca.n_macrostates):
        create_folder('{}/{}'.format(o_dir, macro))
        state_inds = pcca.draw_samples(pccatrajs, n_samples=n_samples)[macro]
        for t_i, (traj_i, frame_i) in enumerate(state_inds):
            print('{}/{}/{:02d}.pdb'.format(o_dir, macro, t_i))
            write_cpptraj_script(
                traj=meta_reset.loc[traj_i]['traj_fn'],
                top=meta_reset.loc[traj_i]['top_fn'],
                frame1=frame_i,
                frame2=frame_i,
                outfile='{}/{}/{:02d}.pdb'.format(o_dir, macro, t_i),
                run=True
            )

        # Get the tICA positions of the selected states
        pcca_frames[macro] = np.asarray(
            [ttrajs_reset[traj_i][frame_i] for traj_i, frame_i in state_inds]
        )
        # Plot these positions on the tICA map
        f, ax = plot_spawns_tica(txx, pcca_frames)
        f.savefig('{}/macrostates_tica_samples.pdf'.format(o_dir))
        plt.close(f)
