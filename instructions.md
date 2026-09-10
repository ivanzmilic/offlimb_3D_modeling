# Modeling off-limb radiation from 3D cubes 

## Goal: Obtain as realistic as possible model of the off-limb emission for the Ca II K line

## Input: 3D structure of a simulated cube + level poupulations of the line 

## Routines: 

extend_cube.ipynb extends a 3D cube to mimic full off-limb LOS 

mpi_synt_cube_simple.py calculates the spectrum, now using a realistic 1D source function (see below) 

calc_op_em.py holds the opacity/emissivity and the source-function table loader + formal solution 

## Source function (implemented): 

The synthesis now uses a real 1D PRD source function instead of the old S=1 / (1-exp(-tau)) assumption. 

- A single total source function S(lambda, z) = eta/chi is read straight from `disk_center_test_opem.fits` (no line/continuum split — continuum is negligible off limb, the core is line-dominated) and applied to the total 3D opacity: eta_3D = chi_3D * S(lambda', z'). 
- `calc_op_em.load_s1d_table(opem_path, wave)` builds the {wave, z, S} table. The FALC table tops out at 2342 km; heights above it are clamped (nearest-edge), not extrapolated. 
- Lookup is velocity-shifted (lambda' = lambda - (v/c)*lambda0, same shift as the Voigt profile) and calibration-shifted (z' = h_ray - dz_cal). 
- `dz_cal` (MURaM z=0 vs FALC z=0 offset, in km) is a free parameter — the 4th CLI arg to `mpi_synt_cube_simple.py`, default 0. **Not yet calibrated** (calibrate by matching disk-center or the limb inflection). 
- The table is loaded once on MPI rank 0 and broadcast to workers; no worker opens the file. 
- `synth(..., s1d, dz_cal, use_source_function=True)` runs the real formal solution when `s1d` is given; the legacy 1-exp(-tau) path is kept behind the flag (s1d=None). 
- Fixed a latent bug in `simple_formal_solution` (was double-counting cumulative transmission, giving I > max(S)); now I = sum_i S_i (1-exp(-dtau_i)) exp(-tau_upwind_i), bounded by max(S). 

## Next step: 

Run the new source-function synthesis on the real cube 
(`/dat/milic/SUSI_modeling/extended_off_limb_ssd_cube_paper_angle_80.h5`) and calibrate `dz_cal`. 
Requires `module load` for openmpi before running (mpi4py import fails in a plain shell). 