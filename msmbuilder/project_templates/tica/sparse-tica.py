"""Reduce dimensionality with Sparse tICA

{{header}}
Meta
----
depends:
  - ftrajs
  - meta.pandas.pickl
"""

from msmbuilder.io import load_trajs, save_generic
from msmbuilder.decomposition import SparseTICA


def do_tica_transform(ftrajs, tica):
    # Fit
    print('Fitting sparse tica...')
    tica.fit(ftrajs.values())
    # Transform
    ttrajs = {}
    for k, v in ftrajs.items():
        print('Transforming traj %s' % meta.iloc[k]['traj_fn'])
        ttrajs[k] = tica.partial_transform(v)
    return tica, ttrajs


# Load
meta, ftrajs = load_trajs("sctrajs")

sptica_list = []
spttrajs_list = []

for rho in [0, 1e-4, 1e-3, 1e-2]:
    tica = SparseTICA(n_components=5, lag_time=10, kinetic_mapping=True, rho=rho)  # 2 ns lag time if 1 stride is used (0.2 ns timestep)
    tica, ttrajs = do_tica_transform(ftrajs, tica)
    sptica_list.append(tica)
    spttrajs_list.append(ttrajs)


# Save
save_generic(sptica_list, 'sptica_list.pkl')
save_generic(spttrajs_list, 'spttrajs_list.pkl')
