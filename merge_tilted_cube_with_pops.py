import numpy as np 
from astropy.io import fits
import h5py

# The goal is to load an h5py file from Paulina obtained by time-x stacking, combine it with level populations calculated by LW 1.5D

import sys

path = sys.argv[1]

atm = sys.argv[2]
pop = sys.argv[3]
final_output = sys.argv[4]

# Load atm
file_atm = h5py.File(path+atm,'r')

# <KeysViewHDF5 ['Electron_density', 'LOS_velocity', 'Pressure', 'Temperature', 'time_steps', 
# 'x_coords_grid', 'x_coords_grid_in_box', 'y_coords_grid', 'y_coords_grid_in_box']>
    
# Load populations
file_pop = fits.open(path+pop)

# Test the shapes. 
print("info::the input atmos quantities have the shape: ", file_atm['Temperature'].shape)
nt, nx, n_step, nz = file_atm['Temperature'].shape

print("info::the input populations have the shape: ", file_pop[2].data.shape)
ntx, n_step, n_lvl, nz = file_pop[2].data.shape

# If this makes sense, we proceed to merge the data:

# Perform the needed reshapes and cuouts to agree with the pops:
T_cutout = np.asarray(file_atm['Temperature']).reshape(nt*nx, n_step, nz)[:ntx,:,:]
P_cutout = np.asarray(file_atm['Pressure']).reshape(nt*nx, n_step, nz)[:ntx,:,:]
Ne_cutout = np.asarray(file_atm['Electron_density']).reshape(nt*nx, n_step, nz)[:ntx,:,:]
V_cutout = np.asarray(file_atm['LOS_velocity']).reshape(nt*nx, n_step, nz)[:ntx,:,:]

# Create some sort of dictionary to save the cube and the corresponding coordinates:
data_to_save = {
    "Temperature": T_cutout,
    "Pressure": P_cutout,
    "Electron_density": Ne_cutout,
    "LOS_velocity": V_cutout,
    "Population_lower_level": file_pop[2].data[:,:,0,::-1],
    "Population_upper_level": file_pop[2].data[:,:,2,::-1],
    "x_coords_grid_in_box": file_atm['x_coords_grid_in_box'],
    "y_coords_grid_in_box": file_atm['y_coords_grid_in_box'],
    "x_coords_grid": file_atm['x_coords_grid'],
    "y_coords_grid": file_atm['y_coords_grid']
}

# And then save to the file:
with h5py.File(path+final_output, 'w') as f:
    for key, value in data_to_save.items():
        f.create_dataset(key, data=value)
