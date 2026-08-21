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

test_wavegrid = np.linspace(391.0, 396.0, 2001)
I, susi_ctx = default_ctx.compute_rays(wavelengths=lw.air_to_vac(test_wavegrid), mus=1.0, returnCtx=True)
susi_ctx.depthData.fill = True
susi_ctx.formal_sol_gamma_matrices()
spec_test = fits.PrimaryHDU(I)
ll_test = fits.ImageHDU(test_wavegrid)
test_hdu = fits.HDUList([spec_test, ll_test])
test_hdu.writeto("disk_center_test.fits", overwrite=True)

        # But this one also needs to spit out the opacities and emissivities for understanding the source function structuring: 
opem = np.zeros((2, len(susi_ctx.atmos.z), len(test_wavegrid)))
opem[0] = np.copy(susi_ctx.depthData.chi[:,0,0,::-1].T)
opem[1] = np.copy(susi_ctx.depthData.eta[:,0,0,::-1].T)
z = np.copy(susi_ctx.atmos.z[::-1])
    
opem_hdu = fits.PrimaryHDU(opem)
z_hdu = fits.ImageHDU(z)
opem_cube = fits.HDUList([opem_hdu, z_hdu])
opem_cube.writeto("disk_center_test_opem.fits", overwrite=True)