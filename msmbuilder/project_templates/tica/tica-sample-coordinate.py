"""Sample tICA coordinates

{{header}}

Meta
----
depends:
  - ../top.pdb
  - ../trajs
"""

import mdtraj as md
from msmbuilder.io.sampling import sample_dimension
from msmbuilder.io import load_trajs, save_generic, preload_tops
from traj_utils import split_trajs_by_type, generate_traj_from_stateinds


def generate_traj(top, inds):
    traj = md.join(
        md.load_frame(meta.loc[traj_i]['traj_fn'], index=frame_i, top=top)
        for traj_i, frame_i in inds
    )
    protein_atoms = traj.top.select('protein')
    traj.superpose(traj,
                   0,
                   atom_indices=protein_atoms)  # superpose to first frame
    return traj


tic = 0
n_samples = 100
scheme = 'linear'

if __name__ == '__main__':
    # Load
    meta, ttrajs = load_trajs('ttrajs')
    ttrajs_subtypes = split_trajs_by_type(ttrajs, meta)

    samples_by_subtype = {key: [] for key in ttrajs_subtypes.keys()}
    print('Found {} types in meta'.format(len(samples_by_subtype.keys())))

    for system in ttrajs_subtypes.keys():
        print('Sampling {} 100 tIC-{} frames'.format(system, tic))
        inds = sample_dimension(ttrajs_subtypes[system],
                                dimension=tic, n_frames=n_samples,
                                scheme=scheme)
        for i, row in meta.iterrows():
            if row['type'] == system:
                for pair in inds:
                    traj_index, frame_index = pair[0], pair[1]
                    if traj_index == i:
                        samples_by_subtype[system].append(pair)

    tops = preload_tops(meta)
    for k, v in samples_by_subtype.items():
        system_name = ''.join(k.split())
        save_generic(v, 'tica-dimension-{}-{}.pkl'.format(tic, system_name))
        top = meta.iloc[v[0][0]]['top_abs_fn']
        tic_traj = generate_traj_from_stateinds(inds=v, meta=meta, atom_selection='not name H1P')
        tic_traj.save_netcdf("tic{}-{}.nc".format(tic, system_name))
