"""Sample a trajectory from microstate MSM

{{header}}

Meta
----
depends:
  - top.pdb
  - trajs
"""

import mdtraj as md
import numpy as np
from msmexplorer import plot_free_energy, plot_trace2d, plot_trace
from msmbuilder.io import load_trajs, save_generic, preload_tops, backup, load_generic
from msmbuilder.io.sampling import sample_msm
from traj_utils import split_trajs_by_type
from matplotlib import pyplot as plt

# Settings
n_steps = 500
st = 1
# Load
meta, ttrajs = load_trajs('ttrajs')
ttrajs_subtypes = split_trajs_by_type(ttrajs, meta)
clusterer = load_generic('clusterer.pkl')
txx = np.concatenate(list(ttrajs.values()))
msms_type = load_generic('msm_dict.pkl')

# Sample
# Warning: make sure ttrajs and clusterer centers have
# the same number of

for system in ttrajs_subtypes.keys():
    system_name = ''.join(system.split())
    msm = msms_type[system]
    ttrajs = ttrajs_subtypes[system]
    inds = sample_msm(ttrajs, clusterer.cluster_centers_, msm, n_steps=n_steps, stride=st)
    save_generic(inds, '{}msm-traj-inds.pkl'.format(system_name))
    ttrajs_of_inds = []
    for pair in inds:
        traj, frame = pair
        ttrajs_of_inds.append(ttrajs[traj][frame])
    # Plot states of traj on tIC space
    f, ax = plt.subplots()
    ax = plot_free_energy(txx, obs=(0, 1), n_samples=10000, ax=ax)
    ax = plot_trace2d(np.asarray(ttrajs_of_inds), cbar_kwargs={'label': 'Frame'})
    ax.set(xlabel='tIC1', ylabel='tIC2', title=str(system))
    f.savefig('{}-msm-traj-tICA-landscape.pdf'.format(system_name))
    # Plot states of traj vs. time
    ax, side_ax = plot_trace(np.asarray(ttrajs_of_inds)[:, 0], label='tIC1')
    plot_trace(np.asarray(ttrajs_of_inds)[:, 1], ax=ax, side_ax=side_ax, color='tarragon', label='tIC2')
    ax.set(xlabel='Frame', ylabel='tIC', title=str(system))
    f = plt.gcf()
    f.tight_layout()
    f.savefig('{}-msm-traj-tICs-frame.pdf'.format(system_name))
    # Make traj
    top = meta.iloc[inds[0][0]]['top_abs_fn']
    traj = md.join(
        md.load_frame(meta.loc[traj_i]['traj_fn'], index=frame_i, top=top)
        for traj_i, frame_i in inds
    )
    traj.center_coordinates()
    traj.superpose(traj, 0)
    # Save traj
    traj_fn = '{}-msm-traj.nc'.format(system_name)
    traj.save_netcdf(traj_fn)
