"""Calculate implied timescales vs. lagtime

{{header}}

Meta
----
depends:
 - meta.pandas.pickl
 - ktrajs
"""
from multiprocessing import Pool

import pandas as pd

from msmbuilder.io import load_trajs, load_meta
from msmbuilder.msm import MarkovStateModel
from functools import partial

# Load
meta = load_meta()
timestep = int(meta['step_ps'].unique())

# Parameters
lagtimes = [
    1,
    10,
    100,
    250,
    500
]
st = 10
print(lagtimes)


## Define what to do for parallel execution
def at_lagtime(lt, ktrajs):
    print('lt = ', lt)
    msm = MarkovStateModel(lag_time=lt, n_timescales=10, verbose=True)
    msm.fit(list(ktrajs.values()))
    ret = {
        'lag_time': lt * timestep * st / 1000,  # in ns
        'percent_retained': msm.percent_retained_,
    }
    for i in range(msm.n_timescales):
        ret['timescale_{}'.format(i)] = msm.timescales_[i] * timestep * st / 1e6  # in microseconds
        ret['timescale_{}_unc'.format(i)] = msm.uncertainty_timescales()[i] * timestep * st / 1e6  # in microseconds
    return ret


def sampled_ktraj(ktraj):
    s_kj = {}
    frames = 0
    for k, v in ktraj.items():
        s_kj[k] = v[::st]
        frames += len(v[::st])
    print('s_kj has {} frames'.format(frames))
    return s_kj


for system in meta.type.unique():
    system_name = ''.join(system.split())
    system_meta = meta[meta.type == system]
    _, ktrajs_subtype = load_trajs('{}_ktrajs'.format(system_name), system_meta)
    with Pool() as p:
        results = p.map(partial(at_lagtime, ktrajs=sampled_ktraj(ktrajs_subtype)), lagtimes)
    df = pd.DataFrame(results)
    # Save
    df.to_pickle('{}timescales.pandas.pkl'.format(system_name))