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
    Based on:
    Voigt function approximation using torch tensors.
    Based on: https://github.com/aasensio/neural_fields/blob/main/utils.py#L174
    """
    A = [122.607931777104326, 214.382388694706425, 181.928533092181549,
         93.155580458138441, 30.180142196210589, 5.912626209773153,
         0.564189583562615]

    B = [122.60793177387535, 352.730625110963558, 457.334478783897737,
         348.703917719495792, 170.354001821091472, 53.992906912940207,
         10.479857114260399, 1.]

    z = damp - np.abs(vv) * 1j

    Z = ((((((A[6] * z + A[5]) * z + A[4]) * z + A[3]) * z + A[2]) * z + A[1]) * z + A[0]) / \
        (((((((z + B[6]) * z + B[5]) * z + B[4]) * z + B[3]) * z + B[2]) * z + B[1]) * z + B[0])

    h = Z.real
    #f = np.sign(vv) * Z.imag * 0.5
    z = vv + damp * 1j
    h = wofz(z).real

    return h/1.7724538509055159 # 1/sqrt(pi)

# The goal is to calculate opacity and emissivity from a 3D cube, that we have precomputed 
# so opposite to before, this is just going to take an array of physical parameters and wavelength

def calc_op_em(param_ray, wavelengths, refine =0, take_given_S=False):

    # param ray contains the necessary physical parameters to solve the RT process:
    T_los = param_ray['Temperature']
    v_los = param_ray['LOS_velocity']
    ne_los = param_ray['Electron_density']
    Pgas_los = param_ray['Pressure']
    pops_l_los = param_ray['Population_lower_level']
    pops_u_los = param_ray['Population_upper_level']
    
    # Make a mask to only take into account the range where T_los is nonzero
    
    mask = T_los > 1.0
    T_los = T_los[mask]
    v_los = v_los[mask]
    ne_los = ne_los[mask]
    Pgas_los = Pgas_los[mask]
    pops_l_los = pops_l_los[mask]
    pops_u_los = pops_u_los[mask]
    
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

    op = np.zeros((len(wavelengths), len(T_los)))
    em = np.zeros((len(wavelengths), len(T_los)))
    
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
    llambda0 = 393.3663E-7 # in cm
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

    # Finally calculate op and em, without the loop:
    op = (const.h.cgs.value * nu0 / (4 * np.pi)) * (pops_l_los[None,:] * B_lu - pops_u_los[None,:] * B_ul) * phi / dnu_D
    #em = op * planck(393.36E-7, T_los)[None,:]
    
    em = (const.h.cgs.value * nu0 / (4 * np.pi)) * pops_u_los[None,:] * A_ul * phi / dnu_D

    # If we want to take the given source function from the population file, we can do that here:
    if (take_given_S):
        opem = fits.open("/home/milic/codes/spherical1d/disk_center_test_opem_muram.fits")[0].data[:,otherids[1],:]
        S = opem[1]/opem[0]
        opemline = opem - opem[:,0][:,None]
        Sline = opemline[1,947]/opemline[0,947]
        em = op * Sline
    
    # Introducing an ad-hoc calculation of the source function to get realistic spectral line.
    #Stemp = pops[4] * A_ul / pops[0] / B_lu
    #print (Stemp[::20])
    
    opc = continuum_opacity(wavelengths[0,None], T_los, ne_los*1E6, nH_los*1E6)/1E2 # in cm^-1
    emc = opc * planck(wavelengths[0]*1E-7, T_los)
    
    op += opc[None,:]
    em += emc[None,:]
    
    #op[T_los<1.0] = 1E-20
    #em[T_los<1.0] = 0.0
        
    return op, em

def simple_formal_solution(op, em, ds):

    dtau = op[:,:] * ds
    tau = np.cumsum(dtau, axis=1)
    Sfn = em / op
    #print (Sfn[300,::10])
    #print(op[300])
    #print(em[300])    
    #exit();
    transmission = np.exp(-tau)
    smalltau = np.where(tau<1E-2)
    transmission[smalltau] = 1.0 - tau[smalltau] + 0.5 * tau[smalltau]**2 - (1.0/6.0) * tau[smalltau]**3
    
    local_contribution = (1.0 - transmission) * Sfn
    local_contribution[smalltau] = dtau[smalltau] * Sfn[smalltau] * (1.0 - 0.5 * dtau[smalltau] + (1.0/6.0) * dtau[smalltau]**2 - (1.0/24.0) * dtau[smalltau]**3)
    # Now integrate over z (axis=1):
    outgoing_contribution = local_contribution * transmission
    contribution_function_noS = transmission * op
    I = np.sum(outgoing_contribution, axis=1)
    return I, tau[:,-1], contribution_function_noS



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