import numpy as np
import matplotlib.pyplot as plt

import lightweaver as lw
import promweaver as pw

import astropy.constants as const
from tqdm import tqdm
from matplotlib.colors import LogNorm
from astropy.io import fits

default_ctx = pw.compute_falc_bc_ctx(active_atoms=["H", "Ca"], prd=True, Nthreads=6)
default_ctx.depthData.fill = True
default_ctx.formal_sol_gamma_matrices()

# Specify the grid in AIR (convenient, matches observations), then convert once to
# the VACUUM frame lightweaver works in. Keep BOTH: wave_vac is what the opacities
# and emissivities are actually computed at, so it is what must be stored alongside
# them (calc_op_em.load_s1d_table reads it back and refuses to guess).

test_wavegrid = np.linspace(391.0, 396.0, 2001)          # AIR wavelengths [nm]
wave_vac = lw.air_to_vac(test_wavegrid)                  # VACUUM wavelengths [nm]
I, susi_ctx = default_ctx.compute_rays(wavelengths=wave_vac, mus=1.0, returnCtx=True)
susi_ctx.depthData.fill = True
susi_ctx.formal_sol_gamma_matrices()
spec_test = fits.PrimaryHDU(I)
ll_test = fits.ImageHDU(test_wavegrid)                   # AIR grid [nm]
ll_test.header['AIRORVAC'] = ('air', 'wavelength frame')
ll_test.header['BUNIT'] = 'nm'
test_hdu = fits.HDUList([spec_test, ll_test])
test_hdu.writeto("disk_center_test.fits", overwrite=True)

# But this one also needs to spit out the opacities and emissivities for understanding the source function structuring:
opem = np.zeros((2, len(susi_ctx.atmos.z), len(wave_vac)))
opem[0] = np.copy(susi_ctx.depthData.chi[:,0,0,::-1].T)
opem[1] = np.copy(susi_ctx.depthData.eta[:,0,0,::-1].T)
z = np.copy(susi_ctx.atmos.z[::-1])

opem_hdu = fits.PrimaryHDU(opem)
opem_hdu.header['AIRORVAC'] = ('vac', 'wavelength frame of the WAVE HDU')
z_hdu = fits.ImageHDU(z)                                  # heights [m]
z_hdu.header['BUNIT'] = 'm'
# Store the wavelength axis explicitly (HDU 2) so load_s1d_table never guesses it.
# This is the VACUUM grid the opacities/emissivities were actually computed at.
wave_hdu = fits.ImageHDU(wave_vac)                       # VACUUM grid [nm]
wave_hdu.header['AIRORVAC'] = ('vac', 'vacuum wavelengths')
wave_hdu.header['BUNIT'] = 'nm'
opem_cube = fits.HDUList([opem_hdu, z_hdu, wave_hdu])
opem_cube.writeto("disk_center_test_opem.fits", overwrite=True)