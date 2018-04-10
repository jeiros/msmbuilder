"""
Plot the validation results
"""
from pyemma.plots import plot_cktest
from msmbuilder.io import load_meta, load_generic
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


# Load
meta = load_meta()
ck_tests = load_generic('ck_tests_pyemma.pkl')
for system in meta.type.unique():
    system_name = ''.join(system.split())
    system_meta = meta[meta.type == system]
    ck = ck_tests[system]

    # Plot the CK results
    f, axarr = plot_cktest(
        ck,
        diag=False,  # Plot every transition
        figsize=(10, 10),
        layout=(ck.nsets, ck.nsets),  # Square reflecting the n x n probs
        padding_top=0.3,  # Leave some space in the top
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

