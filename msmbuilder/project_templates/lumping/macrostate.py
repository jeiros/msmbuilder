"""Make a macrostate MSM

{{header}}
"""

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from msmadapter.utils import create_folder
from traj_utils import write_cpptraj_script, split_trajs_by_type
from plot_utils import figure_dims
import msmexplorer as msme
from msmbuilder.io import load_trajs, load_generic, backup, save_trajs, save_generic
from msmbuilder.lumping import PCCAPlus
from msmbuilder.msm import MarkovStateModel
from msmbuilder.tpt import mfpts
import matplotlib.patches as mpatches

# Settings
n_macrostates = 3
n_samples = 3
plt.style.use('thesis')

# The default colors of UCSF chimera in the order they are assigned
# (at least in my machine) to new models
chimera_colors = [
    '#d2b48c',  # tan
    '#87ceeb',  # sky blue
    '#dda0dd',  # plum
    '#90ee90',  # light green
    '#fa8072',  # salmon
    '#d3d3d3',  # light gray
    '#ff00ff',  # magenta
    '#ffd700',  # gold
    '#1e90ff',  # dodger blue
    '#a020f0',  # purple
]


def reset_keys(a_dict):
    """
    Changes a dictionary keys to be from 0 to N, where N is the number of items
    it has
    """
    new_dict = {}
    i = 0
    for k, v in a_dict.items():
        new_dict[i] = v
        i += 1
    return new_dict


def plot_microstate_mapping(txx, clusterer, pcca):
    # -- COLOR MESS --- #
    # In the followin code I do a dirty mapping of UCSF Chimera default colors
    # to macrostate ID.
    # There is a faster/better/cleaner way to do this I'm sure but can't figure
    # it out atm.
    # Also getting a discrete colorbar in matplotlib is a huge pain in the ass.
    cmap = {}
    for i, c in enumerate(chimera_colors):
        cmap[i] = c
    maping_color = []
    for i in pcca.microstate_mapping_:
        maping_color.append(cmap[i])
    # -- END OF COLOR MESS ---- #

    f, ax = plt.subplots()

    msme.plot_free_energy(
        txx, obs=(0, 1), n_levels=6, vmin=1e-25, ax=ax, alpha=.5,
        cmap='viridis', cbar=False, xlabel='tIC 1', ylabel='tIC 2',

    )

    ax.scatter(
        clusterer.cluster_centers_[msm.state_labels_, 0],
        clusterer.cluster_centers_[msm.state_labels_, 1],
        s=50,
        c=maping_color,

    )
    sns.despine()
    ax.set(xlabel='tIC 1', ylabel='tIC 2')

    # Build legend
    patches = [
        mpatches.Patch(color=chimera_colors[i],
                       label='Macro {}'.format(i))
        for i in range(pcca.n_macrostates)
    ]

    if pcca.n_macrostates <= 3:
        cols = 1
    elif pcca.n_macrostates <= 6:
        cols = 2
    elif pcca.n_macrostates <= 10:
        cols = 3

    plt.legend(
        handles=patches,
        loc='best',
        frameon=True,
        ncol=cols
    )
    return f, ax


def plot_spawns_tica(txx, pcca_frames):
    # -- COLOR MESS --- #
    # In the followin code I do a dirty mapping of UCSF Chimera default colors
    # to macrostate ID.
    # There is a faster/better/cleaner way to do this I'm sure but can't figure
    # it out atm.
    # Also getting a discrete colorbar in matplotlib is a huge pain in the ass.
    cmap = {}
    for i, c in enumerate(chimera_colors):
        cmap[i] = c
    maping_color = []
    for i in pcca.microstate_mapping_:
        maping_color.append(cmap[i])
    # -- END OF COLOR MESS ---- #

    f, ax = plt.subplots()

    msme.plot_free_energy(
        txx, obs=(0, 1), n_levels=6, vmin=1e-25, ax=ax, alpha=.5,
        cmap='viridis', cbar=False, xlabel='tIC 1', ylabel='tIC 2',

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


def plot_microstates(ax, msm, pcca, macro, obs=(0, 1)):
    scale = 100 / np.max(msm.populations_)
    add_a_bit = 50
    prune = clusterer.cluster_centers_[:, obs]
    c = ax.scatter(
        prune[msm.state_labels_, 0],
        prune[msm.state_labels_, 1],
        s=scale * msm.populations_ + add_a_bit,
        c=mfpts(msm, sinks=np.argwhere(pcca.microstate_mapping_ == macro)) * timestep * msm.lag_time
    )
    plt.colorbar(mappable=c, label=r'MFPT ($\mu s$)')
    ax.set_xlabel("tIC 1")
    ax.set_ylabel("tIC 2")
    return ax


if __name__ == '__main__':
    np.random.seed(42)
    # Load
    meta, ttrajs = load_trajs('ttrajs')
    timestep = int(meta['step_ps'].unique()) / 1e6  # ps to us
    clusterer = load_generic('clusterer.pkl')
    txx = np.concatenate(list(ttrajs.values()))
    msms_type = load_generic('msm_dict.pkl')
    ttrajs_subtypes = split_trajs_by_type(ttrajs, meta)

    for system, msm in msms_type.items():
        system_name = ''.join(system.split())
        o_dir0 = '{}'.format(system_name)
        o_dir = '{}/{}_macrostates'.format(system_name, n_macrostates)

        print('--------' + '-' * len(system_name))
        print('Lumping {}'.format(system_name))
        print('--------' + '-' * len(system_name))

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
        save_trajs(pccatrajs, '{}/pccatrajs'.format(o_dir), meta_reset)
        save_generic(pcca, '{}/pcca.pkl'.format(o_dir))

        # Plot microstate mapping to macrostate
        f, ax = plot_microstate_mapping(txx, clusterer, pcca)
        f.suptitle(system)
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
            f.suptitle(system)
            f.savefig('{}/macrostates_tica_samples.pdf'.format(o_dir))
            plt.close(f)
            # Plot a map of the MFPTs from each microstate in the MSM to the set of microstates
            # that belong to this macrostate
            f, ax = plt.subplots(figsize=figure_dims(800, 0.9))
            msme.plot_free_energy(txx, n_levels=5, obs=(0, 1), n_samples=5000, alpha=.1, ax=ax)
            ax = plot_microstates(ax, msm, pcca, macro)
            ax.set_title('Macrostate {} as sink'.format(macro))
            f.tight_layout()
            sns.despine()
            f.savefig('{}/mfpts_to_macro{}.pdf'.format(o_dir, macro))

        # build a coarsed MSM from the pcca trajectories
        # Use double the lag time of the microstate MSM
        # We nedd to use the clip mode in this case
        # The clip mode splits the trajs into smaller ones
        # when there are microstates which do not belong to any
        # macrostate due to the ergodic trimming

        pccatrajs_clipd = pcca.transform(list(ktrajs_subtype.values()), mode='clip')
        msm_coarse = MarkovStateModel(lag_time=msm.lag_time * 2)
        msm_coarse.fit(pccatrajs_clipd)

        # Save the coarsed transition matrix and the stationary distributions
        # of the coarse states
        tmat_fname = '{}/coarsed_tmat.npy'.format(o_dir)
        np.savetxt(tmat_fname, msm_coarse.transmat_, fmt='%4f')
        # Save the stationary distribution
        stat_fname = '{}/stat_dist.npy'.format(o_dir)
        np.savetxt(stat_fname, msm_coarse.left_eigenvectors_[:, 0], fmt='%4f')
