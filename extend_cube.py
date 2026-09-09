import numpy as np 
import matplotlib.pyplot as plt 
from astropy.io import fits
import muram as mio
import sys
import h5py
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator


# First check the original atmosphere and the spectra and make sure they are well-aligned in terms of z (because spectra also contains the populations)

muram_cube = sys.argv[1]
snapno = int(sys.argv[2])
lw_synth = sys.argv[3]
z_offset = int(sys.argv[4])
final_output_file = sys.argv[5]

# Original muram cube:
muram_snap = mio.MuramSnap(muram_cube, snapno)

# Check whether it works.
print (muram_snap.available)

# Shape of the original cube:
print ("info::shape of the original cube: ", muram_snap.Temp.shape)

# Do not transpose anything we will do this later:

# Now check the lw synthesis
lw_output = fits.open(lw_synth)
lw_output.info()

# We only need the pops:
pops = lw_output[2].data
pops.shape

# This underneath assumes that the two cubes are properly aligned and can simply be extended: 

# A quick sanity check with the cube is to see where tau = 1 is: 
# Before everything else check where z = 0, i.e. where is tau = 1 layer
tau_mean = np.mean(muram_snap.tau, axis=(1,2))
z_idx_ph = np.argmin(np.abs(tau_mean - 1))
print("info:: the index of the tau=1 layer is: ", z_idx_ph, "and tau there is = ", tau_mean[z_idx_ph])


# Create a cube that contains all the needed physical quantities: 
# 0 - Temp
# 1 - Pres 
# 2 - n_e 
# 3 - V_los -> calculated by projecting
# 4 - population of the lower level
# 5 - population of the upper level

# This cube has dimensions: N_quantities, N_x, N_steps (number of steps along the line of sight), Nz

# Now we need to create a bigger grid which has various x-start positions, so for each start we have a list of x,y coordinates:
angle = 80  # degrees #TODO: hardcoded at the moment
angle_rad = np.radians(angle)
# starting point in x
x_start = 0
y_start = 0
# these are native steps in x and y
delta_x = 24 # in km # TODO: also hardcoded at the moment, but should be taken from the cube itself
delta_y = 24 # in km
# We want a line that has given length, say - 10ish cubes:
line_length_km = 3E5  # in km # TODO: also hardoced
n_steps = int(line_length_km / delta_x) # number of steps along the line of sight
n_los = pops.shape[0]  # number of lines of sight we want to create, i.e. number of x_start positions
print("number of steps is: ", n_steps)


# Allocate the new cube:
cube_new = np.zeros((6, pops.shape[0], n_steps, pops.shape[3])) # Carefull, this might be huge in memory!
print ("info::shape of the new cube", cube_new.shape)

# Then look a the values at which we need to interpolate:
# For each possible x_start, get the set of values:

x_coords_grid = np.zeros((n_los, n_steps))
y_coords_grid = np.zeros((n_los, n_steps))

x_coords_grid_in_box = np.zeros((n_los, n_steps))
y_coords_grid_in_box = np.zeros((n_los, n_steps))

for i in range(n_los):
    x_start = i * delta_x
    y_start = 0
    x_coords_grid[i, :] = x_start + (np.arange(n_steps) + 0.5) * delta_x * np.cos(angle_rad)
    y_coords_grid[i, :] = y_start + (np.arange(n_steps) + 0.5) * delta_y * np.sin(angle_rad)
    
    # And then fold identically as before:
    x_coords_grid_in_box[i, :] = x_coords_grid[i, :] % ((pops.shape[0]-1)*delta_x)
    y_coords_grid_in_box[i, :] = y_coords_grid[i, :] % ((pops.shape[1]-1)*delta_y)

# Give us a sanity check debug:
print("info::these are the y coordiantes we are interpolating at: ", y_coords_grid_in_box[0])

# ----------------------------------------------------------------------------------------------------------------------
# Finally, the interpolation:

# Loop in z:
#TODO: What would be the easiest way to parallelize this? 

for k in tqdm(range(pops.shape[3])):
    
    interpolator_temp = RegularGridInterpolator((np.arange(pops.shape[0])*delta_x, np.arange(pops.shape[1])*delta_y), muram_snap.Temp[k+z_offset,:,:], bounds_error=False, fill_value=np.nan)
    
    # For each los interpolate:
    for i in range(0,n_los,1):
    
        cube_new[0,i, :, k] = interpolator_temp(np.array([x_coords_grid_in_box[i, :], y_coords_grid_in_box[i, :]]).T).reshape((n_steps,))
    
    interpolator_pres = RegularGridInterpolator((np.arange(pops.shape[0])*delta_x, np.arange(pops.shape[1])*delta_y), muram_snap.Pres[k+z_offset,:,:], bounds_error=False, fill_value=np.nan)
    
    for i in range(n_los):
        cube_new[1,i, :, k] = interpolator_pres(np.array([x_coords_grid_in_box[i, :], y_coords_grid_in_box[i, :]]).T).reshape((n_steps,))
    
    interpolator_ne = RegularGridInterpolator((np.arange(pops.shape[0])*delta_x, np.arange(pops.shape[1])*delta_y), muram_snap.ne[k+z_offset,:,:], bounds_error=False, fill_value=np.nan)
    
    for i in range(n_los):
        cube_new[2,i, :, k] = interpolator_ne(np.array([x_coords_grid_in_box[i, :], y_coords_grid_in_box[i, :]]).T).reshape((n_steps,))
    
    # For V_los we need to calculate it by projecting the velocity vector onto the line of sight. 
    # This is a bit more complicated, but we can do it by first calculating the velocity vector and then projecting it onto the line of sight.
    
    v_los = muram_snap.vy[k+z_offset,:,:] * np.cos(angle_rad) + muram_snap.vz[k+z_offset,:,:] * np.sin(angle_rad)

    # For level populations there is no z_offset.
    
    interpolator_v_los = RegularGridInterpolator((np.arange(pops.shape[0])*delta_x, np.arange(pops.shape[1])*delta_y), v_los, bounds_error=False, fill_value=np.nan)
    for i in range(n_los):
        cube_new[3,i, :, k] = interpolator_v_los(np.array([x_coords_grid_in_box[i, :], y_coords_grid_in_box[i, :]]).T).reshape((n_steps,))    
        
    # Then come the level populations that are also interpolated but from a different array:
    interpolator_pop_lower = RegularGridInterpolator((np.arange(pops.shape[0])*delta_x, np.arange(pops.shape[1])*delta_y), pops[:,:,0,k], bounds_error=False, fill_value=np.nan)
    for i in range(n_los):
        cube_new[4,i, :, k] = interpolator_pop_lower(np.array([x_coords_grid_in_box[i, :], y_coords_grid_in_box[i, :]]).T).reshape((n_steps,))
    
    interpolator_pop_upper = RegularGridInterpolator((np.arange(pops.shape[0])*delta_x, np.arange(pops.shape[1])*delta_y), pops[:,:,1,k], bounds_error=False, fill_value=np.nan)
    for i in range(n_los):
        cube_new[5,i, :, k] = interpolator_pop_upper(np.array([x_coords_grid_in_box[i, :], y_coords_grid_in_box[i, :]]).T).reshape((n_steps,))


# Sanity check plot:
plt.figure(figsize=[14,6])
plt.imshow(cube_new[0,:,:,55], origin="lower", cmap='inferno', aspect='equal',rasterized=True)
plt.colorbar()
plt.savefig("temperature_extension_sanity_check.png", dpi=300,bbox_inches='tight')

# Flip the last two entries because of how optical depth works: 
cube_new[4,:, :, :] = cube_new[4,:, :, ::-1]
cube_new[5,:, :, :] = cube_new[5,:, :, ::-1]

# Create some sort of dictionary to save the cube and the corresponding coordinates:
data_to_save = {
    "Temperature": cube_new[0],
    "Pressure": cube_new[1],
    "Electron_density": cube_new[2],
    "LOS_velocity": cube_new[3],
    "Population_lower_level": cube_new[4],
    "Population_upper_level": cube_new[5],
    "x_coords_grid_in_box": x_coords_grid_in_box,
    "y_coords_grid_in_box": y_coords_grid_in_box,
    "x_coords_grid": x_coords_grid,
    "y_coords_grid": y_coords_grid
}

# And then save to the file:
with h5py.File(final_output_file, 'w') as f:
    for key, value in data_to_save.items():
        f.create_dataset(key, data=value)