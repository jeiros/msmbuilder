"""
Use pyEMMA to build a Bayesian MSM with the same parameters as the microstate.py file
and build a chapman kolmogorov test
"""
from pyemma.msm import BayesianMSM
from pyemma.plots import plot_cktest
from msmbuilder.io import load_trajs, save_generic, load_meta
import datetime
import os
today = datetime.date.today().isoformat()
o_dir = '{}_plots'.format(today)
if not os.path.exists(o_dir):
    os.mkdir(o_dir)

def cleanup_top_right_axes(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return ax

# Settings
lag_time = 100  # 20 ns if a stride of 0.2 ns is used between frames
n_macrostates = 5  # Number of macrostates to do the CK test on
mlags = 6  # Multiple of *MSM* steps to do the CK test for
           # (e.g: if building a 20 ns lag time MSM model, 5 steps is 100 ns, and 6 has to be used since the pyemma function uses a range which goes from 0 to n-1)

# Load
meta = load_meta()
msms_type = {system: None for system in meta.type.unique()}

for system in meta.type.unique():
    system_name = ''.join(system.split())
    system_meta = meta[meta.type == system]
    _, ktrajs_subtype = load_trajs('{}_ktrajs'.format(system_name), system_meta)
    # Fit Bayesian MSM
    msm = BayesianMSM(lag=lag_time, dt_traj='0.2 ns')  # Change dt_traj to match your units
    msm.fit(list(ktrajs_subtype.values()))

    msms_type[system] = msm

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

    # Plot the results
    f, axarr = plot_cktest(
        ck,
        diag=False,  # Plot every transition
        figsize=(10, 10),
        layout=(n_macrostates, n_macrostates),  # Square reflectin the n x n probs
        padding_top=0.2,  # Leave some space in the top
        y01=False, # Do not set all the prob axis from 0 to 1 (since some probs are really small)
        padding_between=0.3,  # Space between sub axes
        # Change this to match your physical time step units between frames in your simulations
        dt=0.2,
        units='ns'
    )
    for col in axarr:
        for ax in col:
            cleanup_top_right_axes(ax)
    f.tight_layout()
    f.savefig('{}/{}ck_test.pdf'.format(o_dir, system_name))

save_generic(msms_type, 'pyemma_msms.pkl')

