# This routine follow the mpi routine we did for 1.5 lw synthesis 
# It takes a atmospheric cube that is already prepared and then splits it pixel by pixel and sytnhesizes the spectrum for each pixel.
# But, the pixels are horizontal tis time, they are not vertical as in the previous routine. 

from threadpoolctl import threadpool_limits

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
threadpool_limits(1)
import pickle
from enum import IntEnum

import numpy as np
from mpi4py import MPI
from tqdm import tqdm
import sys
threadpool_limits(1)

# NOTE(cmo): Numpy please, I beg you, only create 1 BLAS thread per process. 
# NOTE(cmo): Based on Andres' + Andreu's MPI Lightweaver worker

# various i/o stuff:
from astropy.io import fits
import h5py

# And finally physics stuff:
import calc_op_em as coe

# Interpolation:
import scipy.interpolate.RegularGridInterpolator as rgi

# -----------------------------------------------------------------------------------------------------------------------------------

def synth(param_ray, boundary, wavelengths):
    
    # This function synthesises the spectrum for a given atmospheric column.
    op, em = coe.calc_op_em(param_ray, ll, take_given_S=False)
    spectrum_temp, tau, CR = coe.simple_formal_solution(op, op, 24.0e5)
    spectrum = 1.0 - np.exp(-tau)
    return spectrum

# But now we need a function that will create a ray from a given y,z slice, taking into account sphericity:

def create_ray(slice, z_index):
    
    param_ray = 0
    
    # We need to create a grid of y, z values that traverses as it travels in the spherical geometry 
    # Then we bi-linearly interpolate where we can, and where we fall out - we just set zeros (or we don't even need anything)
    
    # Say that max possible lenght of this is N_steps 
    N_steps = 10001
    
    s = (np.arange(N_steps) - N_steps // 2) * 24.0 # km    
    R_at_the_tangent = 696.0E3 + z_index * 24.0 # km
    
    angle = np.arctan(s / R_at_the_tangent)
    y = R_at_the_tangent * angle
    z = z_index * 24.0 + np.sqrt(s**2 + R_at_the_tangent**2) - R_at_the_tangent
    
    # Then we just interpolate the slice to get the parameters at each point.
    y_slice = (np.arange(slice.shape[0]) - slice.shape[0] // 2) * 24.0
    z_slice = (np.arange(slice.shape[1])) * 20.0
    
    # Let's use scipy's regular grid interpolatetor for this
    
    interpolator = rgi((y_slice, z_slice), slice['Temperature'], bounds_error=False, fill_value=0.0)
    T_interpolated = interpolator((y, z))
    interpolator = rgi((y_slice, z_slice), slice['Pressure'], bounds_error=False, fill_value=0.0)
    P_interpolated = interpolator((y, z))
    interpolator = rgi((y_slice, z_slice), slice['Electron_density'], bounds_error=False, fill_value=0.0)
    Ne_interpolated = interpolator((y, z))
    interpolator = rgi((y_slice, z_slice), slice['LOS_velocity'], bounds_error=False, fill_value=0.0)
    V_interpolated = interpolator((y, z))
    interpolator = rgi((y_slice, z_slice), slice['Population_lower_level'], bounds_error=False, fill_value=0.0)
    PopL_interpolated = interpolator((y, z))
    interpolator = rgi((y_slice, z_slice), slice['Population_upper_level'], bounds_error=False, fill_value=0.0)
    PopU_interpolated = interpolator((y, z))
    
    param_ray = {
        'Temperature': T_interpolated,
        'Pressure': P_interpolated,
        'Electron_density': Ne_interpolated,
        'LOS_velocity': V_interpolated,
        'Population_lower_level': PopL_interpolated,
        'Population_upper_level': PopU_interpolated
    }
    
    return param_ray
    

    




