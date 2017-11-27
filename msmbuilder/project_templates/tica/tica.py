"""Reduce dimensionality with tICA

{{header}}
Meta
----
depends:
  - ftrajs
  - meta.pandas.pickl
"""

from msmbuilder.io import load_trajs, save_trajs, save_generic
from msmbuilder.decomposition import tICA

# Load
tica = tICA(n_components=5, lag_time=10, kinetic_mapping=True)
meta, ftrajs = load_trajs("sctrajs")

# Fit
print('Fitting tica...')
tica.fit(ftrajs.values())

# Transform
ttrajs = {}
for k, v in ftrajs.items():
    print('Transforming traj %s' % meta.iloc[k]['traj_fn'])
    ttrajs[k] = tica.partial_transform(v)

# Save
save_trajs(ttrajs, 'ttrajs', meta)
save_generic(tica, 'tica.pkl')
