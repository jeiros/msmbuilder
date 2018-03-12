import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from msmbuilder.io import load_generic, load_trajs, load_meta
from plot_utils import figure_dims, plot_src_sink, plot_microstates, plot_tpt
from traj_utils import split_trajs_by_type, load_in_vmd, \
    get_source_sink, generate_traj_from_stateinds, write_cpptraj_script
from msmadapter.adaptive import create_folder
from msmbuilder.io.sampling import sample_states
from msmbuilder import tpt
import msmexplorer as msme
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
        print(system_name, ev, ev_name)

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
        os.symlink(
            top_path,
            os.path.join('{}/{}/{}'.format(system_name, ev_name, top_basename))
        )
        # Plot source and sink in landscape
        f, ax = plt.subplots(figsize=figure_dims(1800))
        ax = plot_src_sink(
            msm, clusterer, ev, txx, source, sink,
            clabel='{} dynamical eigenvector'.format(ev_name),
            title=system
        )
        f.tight_layout()
        f.savefig(
            os.path.join('{}/{}/landscape.pdf'.format(system_name, ev_name))
        )
        # Frames belonging to source and sinks
        traj_src = generate_traj_from_stateinds(
            src_ev[0], meta,
            atom_selection='not name H1P'
        )
        traj_snk = generate_traj_from_stateinds(
            snk_ev[0], meta,
            atom_selection='not name H1P'
        )

        traj_src.save_netcdf(os.path.join(source_dir, 'clusters.nc'))
        traj_snk.save_netcdf(os.path.join(sink_dir, 'clusters.nc'))
        # Save as PDBs
        i = 0
        for traj_i, frame_id in src_ev[0]:
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
        for traj_i, frame_id in snk_ev[0]:
            cmds = write_cpptraj_script(
                traj=meta.loc[traj_i]['traj_fn'],
                top=meta.loc[traj_i]['top_fn'],
                frame1=frame_id, frame2=frame_id,
                outfile=os.path.join(sink_dir, '{:03d}.pdb'.format(i)),
                write=True,
                run=True
            )
            i += 1

        # TPT analysis
        # Top paths plot
        num_paths = 3

        pos = clusterer.cluster_centers_[msm.state_labels_][:, 0:2]
        w = msm.left_eigenvectors_[:, ev] - msm.left_eigenvectors_[:, ev].min()
        w /= w.max()
        f, ax = plt.subplots(figsize=figure_dims(600, 0.9))
        ax = plot_tpt(
            msm=msm,
            clusterer=clusterer,
            txx=txx,
            ev=ev,
            ax=ax,
            title='Top {} paths -- {} dynamical process'.format(num_paths, ev_name),
            num_paths=num_paths

        )
        f.savefig('{}/{}/top_paths.pdf'.format(system_name, ev_name))
        # Save clusters along top path
        create_folder('{}/{}/top_path'.format(system_name, ev_name))
        net_flux = tpt.net_fluxes(
            [np.argmin(msm.left_eigenvectors_[:, ev])],
            [np.argmax(msm.left_eigenvectors_[:, ev])],
            msm
        )
        paths, _ = tpt.paths(
            [np.argmin(msm.left_eigenvectors_[:, ev])],
            [np.argmax(msm.left_eigenvectors_[:, ev])],
            net_flux,
            num_paths=num_paths)

        for path_no, path in enumerate(paths):
            path_dir = '{}/{}/path{}'.format(system_name, ev_name, path_no)
            create_folder(path_dir)

            states_clusterer_naming = [
                msm.state_labels_[i] for i in path
            ]
            path_inds = sample_states(
                ttrajs_subtypes[system],
                clusterer.cluster_centers_[states_clusterer_naming]
            )
            # Save netcdf trajectory
            path = generate_traj_from_stateinds(
                path_inds, meta, atom_selection='not name H1P'  # To mix S1P and SEP
            )
            path.save_netcdf(os.path.join(path_dir, 'path.nc'))
            # Save PDBs of each frame
            i = 0
            for traj_i, frame_id in path_inds:
                cmds = write_cpptraj_script(
                    traj=meta.loc[traj_i]['traj_fn'],
                    top=meta.loc[traj_i]['top_fn'],
                    frame1=frame_id, frame2=frame_id,
                    outfile=os.path.join(path_dir, '{:03d}.pdb'.format(i)),
                    write=True,
                    run=True
                )
                i += 1
