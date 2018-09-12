"""Cluster tICA results

{{header}}

Meta
----
depends:
 - ttrajs
 - meta.pandas.pickl
"""

from msmbuilder.io import load_trajs, save_trajs, save_generic
from msmbuilder.cluster import MiniBatchKMeans
from traj_utils import split_trajs_by_type

n_clusters = 100

# Load
meta, ttrajs = load_trajs('ttrajs')
ttrajs_subtypes = split_trajs_by_type(ttrajs, meta)
# Fit
clusterer = MiniBatchKMeans(n_clusters=n_clusters)
print('Fitting clustering...')
clusterer.fit([traj for traj in ttrajs.values()])
save_generic(clusterer, 'clusterer.pkl')

# Transform
ktrajs_subtype = {key: None for key in ttrajs_subtypes.keys()}

for system, ttraj in ttrajs_subtypes.items():
    system_name = ''.join(system.split())
    system_meta = meta[meta.type == system]
    # Save
    ktrajs = {}
    for k, v in ttraj.items():
        ktrajs[k] = clusterer.partial_transform(v)

    ktrajs_subtype[system] = ktrajs
    save_trajs(ktrajs_subtype[system], '{}_ktrajs'.format(system_name), system_meta)
