import numpy as np
import matplotlib.pyplot as plt
#import lightweaver as lw
#import promweaver as pw
import astropy.constants as const
from tqdm import tqdm
from matplotlib.colors import LogNorm
from astropy.io import fits
#import xarray as xr
import muram as mio
import scipy.constants as sc
import sys
from scipy.special import wofz

from contop import continuum_opacity

# ---------------------------------------------------------------------------
# Air <-> vacuum wavelength conversion (Morton 2000 coefficients), nm in / nm out.
#
# The whole synthesis runs in VACUUM (lightweaver's frame; llambda0 below and the
# opem source-function grid are both vacuum). Air is only used where wavelengths
# are typed by hand or compared to observations. These two functions are the ONE
# place the conversion is defined -- import them (notebooks included) instead of
# re-deriving, so the two frames can never silently drift apart again.
#
# NOTE the units: sigma must be in 1/micron, i.e. 1e4 / lambda[Angstrom].  A common
# bug is to write 1e4/lambda with lambda in nm, which is 10x too large and makes the
# shift ~10x too small (0.014 nm instead of the correct 0.111 nm at Ca II K).
def _air_refractivity(lam_nm):
    """(n - 1) of standard air at wavelength lam_nm; sigma in 1/micron."""
    sigma2 = (1.0e4 / (np.asarray(lam_nm, dtype=float) * 10.0)) ** 2   # nm -> AA -> (1/micron)
    return (0.00008336624212083
            + 0.02408926869968 / (130.1065924522 - sigma2)
            + 0.0001599740894897 / (38.92568793293 - sigma2))

def air_to_vac(lam_air_nm):
    """Air -> vacuum wavelength [nm]. Ca II K: 393.3663 -> 393.4777 nm."""
    lam = np.asarray(lam_air_nm, dtype=float)
    return lam * (1.0 + _air_refractivity(lam))

def vac_to_air(lam_vac_nm):
    """Vacuum -> air wavelength [nm]; inverse of air_to_vac to sub-pm."""
    lam = np.asarray(lam_vac_nm, dtype=float)
    return lam / (1.0 + _air_refractivity(lam))

def planck(wave, T):
    """
    Planck function in cgs units (erg/s/cm^2/sr/Hz)
    wave: wavelength in cm
    T: temperature in K
    """
    nu = const.c.cgs.value / wave
    c1 = 2.0 * const.h.cgs.value / const.c.cgs.value**2
    c2 = const.h.cgs.value / const.k_B.cgs.value
    B = c1*nu**3 / ((np.exp(c2 *nu / T) - 1.0))
    return B

def fvoigt(damp, vv):
    """
    Voigt function H(a, v) = Re[w(v + i a)] via the Faddeeva function (wofz).
    A rational Humlicek-style approximation used to be computed here and then
    overwritten; since only the wofz value was ever returned, it has been removed.
    """
    z = vv + damp * 1j
    h = wofz(z).real
    return h / 1.7724538509055159  # 1/sqrt(pi)

# The goal is to calculate opacity and emissivity from a 3D cube, that we have precomputed
# so opposite to before, this is just going to take an array of physical parameters and wavelength

def load_s1d_table(opem_path, wave):
    """
    Build the 1D PRD source-function table S(lambda, z) from a precomputed
    disk-center opacity/emissivity file. Meant to be called once (on the MPI
    overseer) and broadcast to the workers.

    A single, total source function is used (no line/continuum split): off limb
    the continuum contributes very little (zero) near the core the line
    dominates the opacity, so S = eta/chi from the file is already the line value.

    Returns a dict with:
      'wave' : wavelength grid [nm], aligned to the synthesis grid `wave`
      'z'    : geometric height grid [km], ascending
      'S'    : total source function, shape [Nwave, Nz]
    """
    hdu = fits.open(opem_path)
    opem = hdu[0].data                            # (2, Nz, Nlam1d): [chi, eta]
    z_1d = np.asarray(hdu[1].data, dtype=float)   # (Nz,) in meters
    # Wavelength axis is read from the file (HDU 2), never guessed. The old
    # hardcoded linspace(392.8, 394.8) has been removed on purpose: generate_s_prd.py
    # now stores the actual VACUUM grid, so a grid/frame mismatch cannot be introduced
    # silently. Regenerate old opem files (without HDU 2) with the current script.
    if len(hdu) < 3 or hdu[2].data is None:
        hdu.close()
        raise ValueError(
            "opem file %r stores no wavelength axis (expected HDU 2). Regenerate it "
            "with the current generate_s_prd.py, which saves the vacuum grid." % opem_path)
    wave_1d = np.asarray(hdu[2].data, dtype=float)          # (Nlam1d,) VACUUM wavelengths [nm]
    frame = str(hdu[2].header.get('AIRORVAC', 'vac')).strip().lower()
    hdu.close()
    if frame.startswith('air'):
        raise ValueError(
            "opem wavelength axis is flagged AIR, but the synthesis works in vacuum "
            "(lightweaver's frame; llambda0 in calc_op_em is vacuum). Store the vacuum "
            "grid instead (see generate_s_prd.py).")

    chi = opem[0].astype(float)                   # (Nz, Nlam1d)
    eta = opem[1].astype(float)
    S = eta / chi                                 # total source function (Nz, Nlam1d)

    # Alignment guard: the opem line core must sit at the vacuum Ca II K rest
    # wavelength (== llambda0 in calc_op_em). If the axis frame or grid is wrong,
    # fail loudly here instead of shifting S under the opacity by ~0.1-0.3 nm silently.
    lambda0_vac = 393.4777                                  # nm; keep in sync with llambda0 in calc_op_em
    k_core = int(np.argmax(chi.max(axis=0)))
    if abs(wave_1d[k_core] - lambda0_vac) > 0.02:
        raise ValueError(
            "opem line core (max opacity) is at %.4f nm but %.4f nm (vacuum Ca II K) was "
            "expected; the wavelength axis looks like the wrong grid or the wrong frame."
            % (wave_1d[k_core], lambda0_vac))

    # Interpolate S onto the synthesis wavelength grid (identity when grids match):
    if chi.shape[1] == len(wave) and np.allclose(wave_1d, wave):
        S_on_wave = S
    else:
        from scipy.interpolate import interp1d
        S_on_wave = interp1d(wave_1d, S, axis=1, bounds_error=False,
                             fill_value="extrapolate")(wave)      # (Nz, Nwave)

    # Heights in km, ascending (RegularGridInterpolator needs increasing grids):
    z_km = z_1d / 1e3
    order = np.argsort(z_km)
    z_km = z_km[order]
    S_final = S_on_wave[order, :]

    return {'wave': np.asarray(wave, dtype=float), 'z': z_km, 'S': S_final}

def calc_op_em(param_ray, wavelengths, refine =0, s1d=None, dz_cal=0.0):

    # param ray contains the necessary physical parameters to solve the RT process:
    T_los = param_ray['Temperature']
    v_los = param_ray['LOS_velocity']
    ne_los = param_ray['Electron_density']
    Pgas_los = param_ray['Pressure']
    pops_l_los = param_ray['Population_lower_level']
    pops_u_los = param_ray['Population_upper_level']
    h_los = param_ray.get('Height', None)   # geometric height along the ray [km], for the S(lambda,z) lookup

    # Make a mask to only take into account the range where T_los is nonzero

    mask = T_los > 1.0
    T_los = T_los[mask]
    v_los = v_los[mask]
    ne_los = ne_los[mask]
    Pgas_los = Pgas_los[mask]
    pops_l_los = pops_l_los[mask]
    pops_u_los = pops_u_los[mask]
    if h_los is not None:
        h_los = h_los[mask]
    
    # Wavelengths are given in nm and later will be converted to cm, to keep working in the infamous cgs 

    # This is a rough approximate for nH_los:
    nH_los = (Pgas_los / (const.k_B.cgs.value * T_los) - ne_los) * 0.9 # 
    
    from scipy.interpolate import interp1d  
    if (refine):
        # Interpolate to a finer grid:
        T_los = interp1d(np.arange(len(T_los)), T_los, kind='cubic')(np.linspace(0,len(T_los)-1,len(T_los)*refine))
        v_los = interp1d(np.arange(len(v_los)), v_los, kind='cubic')(np.linspace(0,len(v_los)-1,len(v_los)*refine))
        ne_los = interp1d(np.arange(len(ne_los)), ne_los, kind='cubic')(np.linspace(0,len(ne_los)-1,len(ne_los)*refine))
        nH_los = interp1d(np.arange(len(nH_los)), nH_los, kind='cubic')(np.linspace(0,len(nH_los)-1,len(nH_los)*refine))
        pops_l_los = interp1d(np.arange(pops_l_los.shape[1]), pops_l_los, kind='cubic', axis=1)(np.linspace(0,pops_l_los.shape[1]-1,pops_l_los.shape[1]*refine))
        pops_u_los = interp1d(np.arange(pops_u_los.shape[1]), pops_u_los, kind='cubic', axis=1)(np.linspace(0,pops_u_los.shape[1]-1,pops_u_los.shape[1]*refine))
        if h_los is not None:
            h_los = interp1d(np.arange(len(h_los)), h_los, kind='cubic')(np.linspace(0,len(h_los)-1,len(h_los)*refine))

    # op and em are assigned in full below; no need to pre-allocate.
    
    # Just to check:    
    '''
    print(T_los)
    print(v_los)
    print(ne_los)
    print(nH_los)
    '''
    # Fix low and high temperatures:
    Tmin = 4000.0
    Tmax = 1E5
    T_los = np.copy(T_los)
    T_los[T_los<Tmin] = Tmin
    T_los[T_los>Tmax] = Tmax

    # Equations for opacity and emissivity:
    # op = (h * nu / 4pi) * (n_l B_lu - n_u B_ul) * phi
    # em = (h * nu / 4pi) * n_u A_ul *
    # where phi is the line profile function (Voigt)
    # for phi we need a and doppler width, recalculated then in frequency units

    # Hard-coded line parameters for now, for Ca II 3933:
    g_l = 2
    g_u = 4
    llambda0 = 393.4777E-7 # in cm; vacuum Ca II K, matches the (vacuum) S(lambda,z) opem table. Was 393.3663E-7 (air), which offset op vs S by 0.1117 nm (the air-vac shift) and made the emergent core red-asymmetric.
    nu0 = const.c.cgs.value / llambda0
    A_ul = 1.47E8
    B_ul = (const.c.cgs.value**2 / (2 * const.h.cgs.value * nu0**3.0)) * A_ul
    B_lu = (g_u / g_l) * B_ul
    gamma = A_ul # natural broadening only
    m_Ca = 40.078 * const.u.cgs.value

    # Doppler velocity:
    dv_D = np.sqrt(2 * const.k_B.cgs.value * T_los / m_Ca)
    # Doppler width in frequency units:
    dl_D = (dv_D / const.c.cgs.value) * llambda0
    dnu_D = (dv_D / const.c.cgs.value) * nu0
    # Damping:
    a = gamma / dnu_D
    # Shifted line center in wavelength units:
    delta_lambda = (v_los / const.c.cgs.value) * llambda0
    # Debug
    
    vv = (wavelengths[:,None]*1E-7 - llambda0 - delta_lambda[None,:]) / dl_D[None,:]

    # Calculate profiles without the loop:
    phi = fvoigt(a[None,:], vv)

    # Line opacity, total, no loop (line + continuum; op needs continuum in both paths):
    op = (const.h.cgs.value * nu0 / (4 * np.pi)) * (pops_l_los[None,:] * B_lu - pops_u_los[None,:] * B_ul) * phi / dnu_D
    opc = continuum_opacity(wavelengths[0,None], T_los, ne_los*1E6, nH_los*1E6)/1E2 # in cm^-1
    op += opc[None,:]

    if s1d is not None and h_los is not None and len(h_los) > 0:
        # Emissivity from the 1D PRD source function: eta = chi * S(lambda', z'),
        # lambda' velocity-shifted (rest frame), z' calibration-shifted. Separable
        # linear interpolation of the (z, lambda) table -- identical result to a 2D
        # RegularGridInterpolator, but without the scattered-point cell search.
        dlam_nm = (v_los / const.c.cgs.value) * llambda0 * 1e7        # cm -> nm
        lam_q = wavelengths[:, None] - dlam_nm[None, :]              # [Nl, Ns]
        h_q = h_los - dz_cal                                        # [Ns], km
        # Clamp to the table range (nearest-edge above the FALC top / beyond the edges):
        lam_q = np.clip(lam_q, s1d['wave'][0], s1d['wave'][-1])
        h_qc = np.clip(h_q, s1d['z'][0], s1d['z'][-1])

        zt = s1d['z']        # ascending heights [Nz1d]
        lw = s1d['wave']     # ascending, uniform wavelength grid [Nlam1d]
        Stab = s1d['S']      # [Nz1d, Nlam1d]

        # (a) linear interpolation in height -> S_z [Ns, Nlam1d]
        iz = np.clip(np.searchsorted(zt, h_qc) - 1, 0, len(zt) - 2)
        wz = np.clip((h_qc - zt[iz]) / (zt[iz + 1] - zt[iz]), 0.0, 1.0)
        S_z = (1.0 - wz)[:, None] * Stab[iz, :] + wz[:, None] * Stab[iz + 1, :]

        # (b) linear interpolation in wavelength (uniform grid) -> S_ray [Nl, Ns]
        dl = lw[1] - lw[0]
        fl = (lam_q - lw[0]) / dl
        il0 = np.clip(np.floor(fl).astype(np.intp), 0, len(lw) - 2)
        wl = np.clip(fl - il0, 0.0, 1.0)
        s_idx = np.arange(h_qc.shape[0])[None, :]                   # [1, Ns]
        S_ray = (1.0 - wl) * S_z[s_idx, il0] + wl * S_z[s_idx, il0 + 1]

        em = op * S_ray
    else:
        # Legacy emissivity: spontaneous line term + continuum emissivity.
        em = (const.h.cgs.value * nu0 / (4 * np.pi)) * pops_u_los[None,:] * A_ul * phi / dnu_D
        emc = opc * planck(wavelengths[0]*1E-7, T_los)
        em += emc[None,:]

    return op, em

def simple_formal_solution(op, em, ds):

    # Discrete formal solution, observer at index 0, ray running along increasing index:
    #   I = sum_i S_i * (1 - exp(-dtau_i)) * exp(-tau_upwind_i)
    # where dtau_i is the cell's own optical depth and tau_upwind_i is the optical depth
    # from the observer to the near edge of the cell. This is bounded by max(S), as it must be.
    dtau = op * ds
    tau = np.cumsum(dtau, axis=1)     # optical depth at the far edge of each cell
    tau_upwind = tau - dtau           # optical depth from the observer to the near edge
    Sfn = em / op

    escape = -np.expm1(-dtau)         # 1 - exp(-dtau), stable as dtau -> 0
    weight = escape * np.exp(-tau_upwind)   # contribution function per cell (no S)
    outgoing_contribution = weight * Sfn
    I = np.sum(outgoing_contribution, axis=1)
    return I, tau[:, -1], weight



if __name__=='__main__':
    pops_file= sys.argv[1]
    path_to_muram = sys.argv[2]
    snapshot_id = int(sys.argv[3])
    i = int(sys.argv[4])

    # Smaller range for testing:
    wavelengths = np.linspace(393.06, 393.66, 601)

    refine = 2

    ktest = 192
    
    op, em = calc_op_em(pops_file, path_to_muram, snapshot_id, wavelengths, axis=1, otherids=(i, 192), refine=refine)

    # Now let's do a simple formal solution:
    ds = 24e5 # in cm, MURaM grid spacing
    I, tau_los, temp = simple_formal_solution(op, em, ds/refine)
    print(tau_los.shape)

    # And plot the results, to test:
    plt.figure(figsize=(10,6))
    I_proxy = 1.0 - np.exp(-tau_los)
    plt.plot(wavelengths, I_proxy)
    plt.savefig("figs/"+str(i)+"_"+str(ktest)+"_test_off_limb.png")
    #exit();


    # And now the full slit:
    # actually repeat for multiple slits: 

    for i in range(0,8):
        I_slit = np.zeros((401, len(wavelengths)))
        tau_los = np.zeros((401, len(wavelengths)))
        outgoing_contribution = np.zeros([401, len(wavelengths), 1024*1])
        for k in tqdm(range(0,401)):
            op, em = calc_op_em(pops_file, path_to_muram, snapshot_id, wavelengths, axis=1, otherids=(i, k), refine=0, take_given_S=False)
            I_slit[k,:], tau_los[k,:], outgoing_contribution[k,:,:] = simple_formal_solution(op, em, ds/refine)

        kek = fits.PrimaryHDU(I_slit)
        bur = fits.ImageHDU(tau_los)
        bur2 = fits.ImageHDU(outgoing_contribution)
        lol = fits.HDUList([kek, bur, bur2])
        lol.writeto("/dat/milic/SUSI_modeling/"+str(i)+"_test_off_limb_che_slit.fits",overwrite=True)