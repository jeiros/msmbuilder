"""
Plot the validation results
"""
import datetime
import os

import seaborn as sns
from pyemma._base.model import SampledModel
from pyemma.msm import ChapmanKolmogorovValidator
from pyemma.plots import plot_cktest

from msmbuilder.io import load_meta

sns.set_style('ticks')


today = datetime.date.today().isoformat()
o_dir = '{}_plots'.format(today)
if not os.path.exists(o_dir):
    os.mkdir(o_dir)


# Load
meta = load_meta()
for system in meta.type.unique():
    system_name = ''.join(system.split())
    system_meta = meta[meta.type == system]
    ck = ChapmanKolmogorovValidator.load('ck_tests_pyemma.pkl', system_name)
    ck.has_errors = issubclass(ck.test_model.__class__, SampledModel)

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
    sns.despine()
    f.tight_layout()
    f.savefig('{}/{}ck_test.pdf'.format(o_dir, system_name))
