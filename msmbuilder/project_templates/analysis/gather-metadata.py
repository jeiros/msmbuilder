"""Find trajectories and associated metadata

{{header}}

Meta
----
depends:
  - trajs
  - top.pdb
"""

from msmbuilder.io import gather_metadata, save_meta, NumberedRunsParser, render_meta
import pandas as pd

# Construct and save the dataframe
WT_apo_uP_parser = NumberedRunsParser(
    traj_fmt="run-{run}_imaged_superposed.nc",
    top_fn="WT_uP_apo.prmtop",
    step_ps=200
)
WT_apo_S1P_parser = NumberedRunsParser(
    traj_fmt="S1P-traj-{run}_imaged_superposed.nc",
    top_fn="WT_S1P_apo.prmtop",
    step_ps=200
)
WT_apo_SEP_parser = NumberedRunsParser(
    traj_fmt="SEP-traj-{run}_imaged_superposed.nc",
    top_fn="WT_SEP_apo.prmtop",
    step_ps=200
)
G159D_SilA_S1P_parser = NumberedRunsParser(
    traj_fmt="run{run}_filt_sup.nc",
    top_fn="G159D_S1P_SilA.prmtop",
    step_ps=200
)
G159D_SilB_S1P_parser = NumberedRunsParser(
    traj_fmt="run{run}_sup_filt.nc",
    top_fn="G159D_S1P_SilB.prmtop",
    step_ps=200
)
WT_apo_uP_meta = gather_metadata("trajs/WT/apo/uP/*.nc", WT_apo_uP_parser)
WT_apo_S1P_meta = gather_metadata("trajs/WT/apo/S1P/*.nc", WT_apo_S1P_parser)
WT_apo_SEP_meta = gather_metadata("trajs/WT/apo/SEP/*.nc", WT_apo_SEP_parser)
G159D_SilA_S1P_meta = gather_metadata("trajs/G159D/SilybinA/S1P/*nc", G159D_SilA_S1P_parser)
G159D_SilB_S1P_meta = gather_metadata("trajs/G159D/SilybinB/S1P/*nc", G159D_SilB_S1P_parser)
# Add column to metadata with string representation of system, for clarity
WT_apo_uP_meta['type'] = 'WT apo uP'
WT_apo_S1P_meta['type'] = 'WT apo S1P'
WT_apo_SEP_meta['type'] = 'WT apo SEP'
G159D_SilA_S1P_meta['type'] = 'G159D SilybinA S1P'
G159D_SilB_S1P_meta['type'] = 'G159D SilybinB S1P'

# All
meta = pd.concat([WT_apo_uP_meta, WT_apo_S1P_meta,
                  WT_apo_SEP_meta, G159D_SilA_S1P_meta, G159D_SilB_S1P_meta])
meta = meta.reset_index(drop=True)
meta.index.name = 'run'
save_meta(meta)
render_meta(meta, title='metadata')
# WT systems meta
meta_WT = pd.concat([WT_apo_uP_meta, WT_apo_S1P_meta,
                     WT_apo_SEP_meta])
meta_WT = meta_WT.reset_index(drop=True)
meta_WT.index.name = 'run'
save_meta(meta_WT, 'WT_meta.pickl')
render_meta(meta_WT, 'WT_meta.html')

# G159D systems meta
meta_G159D = pd.concat([G159D_SilA_S1P_meta, G159D_SilB_S1P_meta])
meta_G159D = meta_G159D.reset_index(drop=True)
meta_G159D.index.name = 'run'
save_meta(meta_G159D, 'G159D_meta.pickl')
render_meta(meta_G159D, 'G159D_meta.html')

# Individual metadata objects
# WT apo uP
save_meta(WT_apo_uP_meta, 'WT_apo_uP_meta.pickl')
render_meta(WT_apo_uP_meta, 'WT_apo_uP_meta.html')
# WT apo S1P
save_meta(WT_apo_S1P_meta, 'WT_apo_S1P_meta.pickl')
render_meta(WT_apo_S1P_meta, 'WT_apo_S1P_meta.html')
# WT apo SEP
save_meta(WT_apo_SEP_meta, 'WT_apo_SEP_meta.pickl')
render_meta(WT_apo_SEP_meta, 'WT_apo_SEP_meta.html')
# G159D SilybinA S1P
save_meta(G159D_SilA_S1P_meta, 'G159D_SilA_S1P_meta.pickl')
render_meta(G159D_SilA_S1P_meta, 'G159D_SilA_S1P_meta.html')
# G159D SilybinB S1P
save_meta(G159D_SilB_S1P_meta, 'G159D_SilB_S1P_meta.pickl')
render_meta(G159D_SilB_S1P_meta, 'G159D_SilB_S1P_meta.html')
