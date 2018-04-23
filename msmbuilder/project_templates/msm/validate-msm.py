"""
Use pyEMMA to build a Bayesian MSM with the same parameters as the microstate.py file
and build a chapman kolmogorov test
"""

from pyemma.msm import BayesianMSM

from msmbuilder.io import load_meta, load_trajs

# Settings
lag_time = 100  # 20 ns if a stride of 0.2 ns is used between frames
n_macrostates = 5  # Number of macrostates to do the CK test on
mlags = 6  # Multiple of *MSM* steps to do the CK test for
           # (e.g: if building a 20 ns lag time MSM model, 5 steps is 100 ns, and 6 has to be used since the pyemma function uses a range which goes from 0 to n-1)


# Load
meta = load_meta()
for system in meta.type.unique():
    system_name = ''.join(system.split())
    system_meta = meta[meta.type == system]
    _, ktrajs_subtype = load_trajs('{}_ktrajs'.format(system_name), system_meta)
    # Fit Bayesian MSM
    msm = BayesianMSM(lag=lag_time, dt_traj='0.2 ns')  # Change dt_traj to match your units
    msm.fit(list(ktrajs_subtype.values()))

    # Do CK test
    print('-----------------------' + '-'*len(system_name))
    print('Performing CK test for {}'.format(system_name))
    print('-----------------------' + '-' * len(system_name))
    ck = msm.cktest(
        n_macrostates,
        err_est=True,
        n_jobs=-1,
        mlags=mlags
    )
    ck.save('ck_tests_pyemma.pkl', system_name)
