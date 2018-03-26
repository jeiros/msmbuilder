"""Make a microstate MSM

{{header}}
"""

from msmbuilder.io import load_trajs, save_trajs, save_generic, load_meta
from msmbuilder.msm import MarkovStateModel


# Settings
lag_time = 1
n_timescales = 5

# Load
meta = load_meta()
msms_type = {system: None for system in meta.type.unique()}
for system in meta.type.unique():
    system_name = ''.join(system.split())
    system_meta = meta[meta.type == system]
    _, ktrajs_subtype = load_trajs('{}_ktrajs'.format(system_name), system_meta)
    # Fit MSM
    msm = MarkovStateModel(lag_time=lag_time, n_timescales=n_timescales, verbose=True)
    msm.fit(list(ktrajs_subtype.values()))

    msms_type[system] = msm
    # Transform
    microktrajs = {}
    for k, v in ktrajs_subtype.items():
        microktrajs[k] = msm.partial_transform(v)

    # Save
    save_generic(msm, '{}_msm.pkl'.format(system_name))
    save_trajs(microktrajs, '{}_microktrajs'.format(system_name), system_meta)

# Save all MSMs together
save_generic(msms_type, 'msm_dict.pkl')
