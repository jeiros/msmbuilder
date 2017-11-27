"""Generate PDB files for sources and sinks

{{header}}
"""


import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from msmbuilder.io import load_generic, load_trajs
from plot_utils import figure_dims, plot_src_sink, plot_microstates
from traj_utils import split_trajs_by_type, load_in_vmd, \
    get_source_sink, generate_traj_from_stateinds, write_cpptraj_script
from msmbuilder.io.sampling import sample_states
import os
sns.set_style('ticks')
colors = sns.color_palette()
n_clusters = 10

# Load
meta, ttrajs = load_trajs('ttrajs')
clusterer = load_generic('clusterer.pkl')
txx = np.concatenate(list(ttrajs.values()))
msms_type = load_generic('msm_dict.pkl')
ttrajs_subtypes = split_trajs_by_type(ttrajs, meta)

for system, msm in msms_type.items():
    system_name = ''.join(system.split())
    for ev, ev_name in zip(range(1, 4), ['1st', '2nd', '3rd']):
        source, sink = get_source_sink(msm, clusterer, eigenvector=ev)
        # Define and create folders
        source_dir = '{}/{}/source'.format(system_name, ev_name)
        sink_dir = '{}/{}/sink'.format(system_name, ev_name)
        if not os.path.isdir(source_dir):
            os.makedirs(source_dir)
        if not os.path.isdir(sink_dir):
            os.makedirs(sink_dir)
        # Find inds of samples
        src_ev = sample_states(
            ttrajs_subtypes[system],
            clusterer.cluster_centers_[[source]],
            k=n_clusters
        )
        snk_ev = sample_states(
            ttrajs_subtypes[system],
            clusterer.cluster_centers_[[sink]],
            k=n_clusters
        )
        # Add symlink of prmtop for easy loading after with Chimera
        a_traj_index = src_ev[0][0][0]  # anyone will do since they are already split by traj type at this point
        top_path = meta.loc[a_traj_index]['top_abs_fn']
        top_basename = os.path.basename(top_path)
        os.symlink(top_path, os.path.join('{}/{}/{}'.format(system_name, ev_name, top_basename)))
        # Plot source and sink in landscape
        f, ax = plt.subplots(figsize=figure_dims(1800))
        ax = plot_src_sink(
            msm, clusterer, ev, txx, source, sink,
            clabel='{} dynamical eigenvector'.format(ev_name),
            title=system
        )
        f.tight_layout()
        f.savefig(os.path.join('{}/{}/{}.pdf'.format(system_name, ev_name, 'landscape')))
        # Frames belonging to source and sinks
        traj_src = generate_traj_from_stateinds(src_ev, meta)
        traj_snk = generate_traj_from_stateinds(snk_ev, meta)

        traj_src.save_netcdf(os.path.join(source_dir, 'clusters.nc'))
        traj_snk.save_netcdf(os.path.join(sink_dir, 'clusters.nc'))
        # Save as PDBs
        i = 0
        for traj_i, frame_id in src_ev[0][0: int(len(src_ev[0]) / 2)]:
            cmds = write_cpptraj_script(
                traj=meta.loc[traj_i]['traj_fn'],
                top=meta.loc[traj_i]['top_fn'],
                frame1=frame_id, frame2=frame_id,
                outfile=os.path.join(source_dir, '{:03d}.pdb'.format(i)),
                write=True,
                run=True
            )
            i += 1

        i = 0
        for traj_i, frame_id in snk_ev[0][0: int(len(snk_ev[0]) / 2)]:
            cmds = write_cpptraj_script(
                traj=meta.loc[traj_i]['traj_fn'],
                top=meta.loc[traj_i]['top_fn'],
                frame1=frame_id, frame2=frame_id,
                outfile=os.path.join(sink_dir, '{:03d}.pdb'.format(i)),
                write=True,
                run=True
            )
            i += 1
