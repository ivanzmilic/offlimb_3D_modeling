# This routine follow the mpi routine we did for 1.5 lw synthesis 
# It takes a atmospheric cube that is already prepared and then splits it pixel by pixel and sytnhesizes the spectrum for each pixel.
# But, the pixels are horizontal this time, they are not vertical as in the previous routine. 

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

# Interpolation: create_ray now uses precomputed bilinear weights (see _ray_weights),
# so RegularGridInterpolator is no longer needed here.

# -----------------------------------------------------------------------------------------------------------------------------------

def airtovac(lambda_air):
    # Backward-compatible shim. The canonical air<->vacuum conversion now lives in
    # calc_op_em (coe.air_to_vac / coe.vac_to_air) so it is defined in exactly one place.
    return coe.air_to_vac(lambda_air)

def synth(param_ray, wavelengths, ds, boundary=0, s1d=None, dz_cal=0.0, use_source_function=False):

    # This function synthesises the spectrum for a given atmospheric column.
    # ds is the geometric step along the ray [cm] (should equal delta_s * 1e5).

    op, em = coe.calc_op_em(param_ray, wavelengths, s1d=s1d, dz_cal=dz_cal)

    if use_source_function and s1d is not None:
        # Real formal solution with the 1D PRD source function S(lambda, z):
        I, tau, CR = coe.simple_formal_solution(op, em, ds)
        return I
    else:
        # Legacy behaviour: constant source function (S = 1), so I = 1 - exp(-tau).
        spectrum_temp, tau, CR = coe.simple_formal_solution(op, op, ds)
        return 1.0 - np.exp(-tau)

# But now we need a function that will create a ray from a given y,z slice, taking into account sphericity:

def create_curved_grid(z_index, delta_s=24.0, delta_z=20.0, R_sun=696.0E3):

    # Meant for testing, but we can also just call it from create ray and then interpolate the slice to get the parameters at each point.
    # delta_s : sampling step along the LOS ray [km]
    # delta_z : vertical grid spacing of the cube [km]
    # R_sun   : solar radius [km]
    N_steps = 10001

    s = (np.arange(N_steps) - N_steps // 2) * delta_s # km
    R_at_the_tangent = R_sun + z_index * delta_z # km

    angle = np.arctan(s / R_at_the_tangent)
    y = R_at_the_tangent * angle
    z = z_index * delta_z + np.sqrt(s**2 + R_at_the_tangent**2) - R_at_the_tangent

    return s, y, z

# Precomputed bilinear weights mapping the (NY, NZ) slit plane onto ray z_index's
# curved-ray points. The geometry depends only on z_index (and the fixed plane grid),
# so the weights are identical across all slits -- compute once per z_index and cache.
_RAY_W_CACHE = {}

def _ray_weights(z_index, NY, NZ, delta_los=24.0, delta_z=20.0):
    key = (z_index, NY, NZ)
    cached = _RAY_W_CACHE.get(key)
    if cached is not None:
        return cached
    s, y, z = create_curved_grid(z_index)
    # Fractional grid indices of the ray points on the plane grids:
    #   y_slice = (arange(NY) - NY//2) * delta_los  ->  fy = y/delta_los + NY//2
    #   z_slice =  arange(NZ)          * delta_z    ->  fz = z/delta_z
    fy = y / delta_los + (NY // 2)
    fz = z / delta_z
    # In-box test (reproduces rgi bounds_error=False, fill_value=0.0):
    valid = (fy >= 0) & (fy <= NY - 1) & (fz >= 0) & (fz <= NZ - 1)
    iy0 = np.clip(np.floor(fy).astype(np.intp), 0, NY - 2)
    iz0 = np.clip(np.floor(fz).astype(np.intp), 0, NZ - 2)
    wy = fy - iy0
    wz = fz - iz0
    # Flat indices into a C-order (NY, NZ) plane:
    i00 = (iy0 * NZ + iz0).astype(np.int32)
    i01 = (iy0 * NZ + (iz0 + 1)).astype(np.int32)
    i10 = ((iy0 + 1) * NZ + iz0).astype(np.int32)
    i11 = ((iy0 + 1) * NZ + (iz0 + 1)).astype(np.int32)
    # Corner weights, zeroed outside the box so out-of-grid points evaluate to 0:
    w00 = np.where(valid, (1.0 - wy) * (1.0 - wz), 0.0)
    w01 = np.where(valid, (1.0 - wy) * wz, 0.0)
    w10 = np.where(valid, wy * (1.0 - wz), 0.0)
    w11 = np.where(valid, wy * wz, 0.0)
    out = (i00, i01, i10, i11, w00, w01, w10, w11, z)
    _RAY_W_CACHE[key] = out
    return out

def create_ray(slice, z_index):

    # Bilinearly interpolate the (NY, NZ) slit plane onto the curved ray, using
    # precomputed weights -- same result as the old per-quantity RegularGridInterpolator
    # calls, but built once per z_index and reused across every slit.
    NY = slice['Temperature'].shape[0]
    NZ = slice['Temperature'].shape[1]
    i00, i01, i10, i11, w00, w01, w10, w11, z = _ray_weights(z_index, NY, NZ)

    def gather(Q):
        f = np.asarray(Q).reshape(-1)
        return f[i00] * w00 + f[i01] * w01 + f[i10] * w10 + f[i11] * w11

    param_ray = {
        'Temperature': gather(slice['Temperature']),
        'Pressure': gather(slice['Pressure']),
        'Electron_density': gather(slice['Electron_density']),
        'LOS_velocity': gather(slice['LOS_velocity']),
        'Population_lower_level': gather(slice['Population_lower_level']) / 1E6, # to cgs
        'Population_upper_level': gather(slice['Population_upper_level']) / 1E6,
        'Height': z # geometric height above the surface along the curved ray [km], for S(lambda,z)
    }

    return param_ray

class tags(IntEnum):
    """ Class to define the state of a worker.
    It inherits from the IntEnum class """ # Makes sense to me, but not sure what is the IntEnum class 
    READY = 0
    DONE = 1
    EXIT = 2
    START = 3
    
def slice_tasks(cube, task_start, grain_size):
    
    task_end = min(task_start + grain_size, cube['Temperature'].shape[0])

    
    sl = slice(task_start, task_end) # this is a slice object, allowing us to access the specific thingy
    
    data = {}
    data['taskGrainSize'] = task_end - task_start

    data['Temperature'] = cube['Temperature'][sl,:,:]
    data['Pressure'] =          cube['Pressure'][sl,:,:]
    data['Electron_density'] = cube['Electron_density'][sl,:,:]
    data['LOS_velocity'] =        cube['LOS_velocity'][sl,:,:]
    data['Population_lower_level'] = cube['Population_lower_level'][sl,:,:]
    data['Population_upper_level'] = cube['Population_upper_level'][sl,:,:]

    return data

def overseer_work(cube, wave, task_grain_size=16, end=None, task_info=None):
    
    """ Function to define the work to do by the overseer """

    # Reshape the atmosphere:
    NX,NY,NZ = cube["Temperature"].shape
    
    # Index of the task to keep track of each job
    task_index = 0
    num_workers = size - 1
    closed_workers = 0

    # TODO: Figure out these comments so one can know what it all means
    data_size = 0 # Let's figure out what this is - total number of pixels?
    num_tasks = 0 # And this is data_size // 16? 
    file_idx_for_task = [] # does this have sth to do with reading from file?
    task_start_idx = [] # no idea
    task_writeback_range = [] # no idea
    
    cdf_size = cube["Temperature"].shape[0] if end is None else end
    print("info::overseer::cdf_size = ", cdf_size)

    num_cdf_tasks = int(np.ceil(cdf_size / task_grain_size)) # number of tasks = roundedup number of pixels / grain
    
    task_start_idx.extend(range(0, cdf_size, task_grain_size))
    
    task_writeback_range.extend([slice(data_size + i*task_grain_size, min(data_size + (i+1)*task_grain_size,
        data_size + cdf_size)) for i in range(num_cdf_tasks)])
    
    data_size = cdf_size
    num_tasks = num_cdf_tasks

    # Define the lists that will store the data of each feature-label pair - I hate lists, can I work with 
    # numpy array 
    slits = [None] * data_size
    
    success = True
    task_status = [0] * num_tasks

    with tqdm(total=num_tasks, ncols=110) as progress_bar:
        
        # While we don't have more closed workers than total workers keep looping
        while closed_workers < num_workers:
            data_in = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag = status.Get_tag()

            if tag == tags.READY:
                try:
                    task_index = task_status.index(0)
                    
                    # Slice out our task
                    data = slice_tasks(cube, task_start_idx[task_index], task_grain_size)
                    data['index'] = task_index
                    data['wave'] = wave
                    
                    # send the data of the task and put the status to 1 (done)
                    comm.send(data, dest=source, tag=tags.START)
                    task_status[task_index] = 1

                # If error, or no work left, kill the worker
                except:
                    comm.send(None, dest=source, tag=tags.EXIT)

            # If the tag is Done, receive the status, the index and all the data
            # and update the progress bar
            elif tag == tags.DONE:
                success = data_in['success']
                task_index = data_in['index']

                if not success:
                    task_status[task_index] = 0
                    print(f"Task: {task_index} failed")
                else:
                    task_writeback = task_writeback_range[task_index]
                    slits[task_writeback] = data_in['slits']
                    progress_bar.update(1)

            # if the worker has the exit tag mark it as closed.
            elif tag == tags.EXIT:
                #print(" * Overseer : worker {0} exited.".format(source))
                closed_workers += 1

    # Once finished, dump all the data
    slits = np.asarray(slits)
    
    spechdu = fits.PrimaryHDU(slits)
    wavhdu = fits.ImageHDU(wave)
    to_output = fits.HDUList([spechdu, wavhdu])
    if (task_info is not None):
        path, filename, number = task_info
        to_output.writeto(path[:-3]+filename+'_'+str(number)+'.fits', overwrite=True)    
    else:
        to_output.writeto('/dat/milic/offlimb_output_new_sliced.fits', overwrite=True)

    return 0  

def worker_work(rank, s1d=None, dz_cal=0.0):
    # Function to define the work that the workers will do

    while True:
        # Send the overseer the signal that the worker is ready
        comm.send(None, dest=0, tag=tags.READY)
        # Receive the data with the index of the task, the atmosphere parameters and/or the tag
        data_in = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()

        if tag == tags.START:
            # Receive the y,z slice
            task_index = data_in['index'] # I think we need this? - for what though (to keep track of what succeeeded where)
            Temperature = data_in['Temperature'].astype(float)
            Pressure = data_in['Pressure'].astype(float)
            Electron_density = data_in['Electron_density'].astype(float)
            LOS_velocity = data_in['LOS_velocity'].astype(float)
            lower_level_population = data_in['Population_lower_level'].astype(float)
            upper_level_population = data_in['Population_upper_level'].astype(float)
            
            task_size = data_in['taskGrainSize']
            wave = data_in['wave'].astype(float)
            
            slit_height = Temperature.shape[2]
            
            I = np.zeros([task_size, slit_height, len(wave)])
            
            for i in range(task_size):
                
                # Make a slice:
                param_slice = {
                    'Temperature': Temperature[i,:,:],
                    'Pressure': Pressure[i,:,:],
                    'Electron_density': Electron_density[i,:,:],
                    'LOS_velocity': LOS_velocity[i,:,:],
                    'Population_lower_level': lower_level_population[i,:,:],
                    'Population_upper_level': upper_level_population[i,:,:]
                }
                #for j in tqdm(range(slit_height)):
                for j in range(slit_height):
                    
                    # Use the functions to create the ray and then synthesize the spectrum for this ray
                    param_ray = create_ray(param_slice, j)

                    # ds = delta_s (24 km) * 1e5 -> cm; keep in sync with create_curved_grid's delta_s
                    I[i,j,:] = synth(param_ray, wavelengths=wave, ds=24.0e5, boundary=0, s1d=s1d, dz_cal=dz_cal, use_source_function=True)
            
            success = 1
            

            # Send the computed data
            # we do want to fill in tau too, but that can wait for the next step
            data_out =  {'index': task_index, 'success': success, 'slits': I}
            comm.send(data_out, dest=0, tag=tags.DONE)

        # If the tag is exit break the loop and kill the worker and send the EXIT tag to overseer
        elif tag == tags.EXIT:
            break

    comm.send(None, dest=0, tag=tags.EXIT)
    
    
if (__name__ == '__main__'):

    # Initializations and preliminaries
    comm = MPI.COMM_WORLD   # get MPI communicator object
    size = comm.size        # total number of processes
    rank = comm.rank        # rank of this process
    status = MPI.Status()   # get MPI status object
    
    #print(f"Node {rank}/{size} active", flush=True)

    if rank == 0: # If I am the overseer process

        print("info::overseer::starting...")

        # --------------------------------------------------------------------
        path = sys.argv[1] # path where the data is
        end = int(sys.argv[2])
        filename = sys.argv[3]
        dz_cal = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0  # MURaM z=0 vs FALC z=0 offset [km]
        number = 0

        #stokes = sys.argv[5].lower() == 'true' # whether to synthesize Stokes I or all 4 components

        cube = h5py.File(path,'r')

        wave = np.linspace(392.8,394.8,1001)

        # Build the 1D PRD source-function table once, here on the overseer, and broadcast below.
        # (Only rank 0 opens the file on disk.)
        opem_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "disk_center_test_opem.fits")
        s1d = coe.load_s1d_table(opem_path, wave)

        print("info::overseer::input cube shape is: ", cube['Temperature'].shape)
        print("info::overseer::loaded 1D source function table, S shape =", s1d['S'].shape,
              ", dz_cal =", dz_cal, "km")
    else:
        s1d = None
        dz_cal = None

    # Collective broadcast of the source-function table and calibration offset to every worker.
    s1d = comm.bcast(s1d, root=0)
    dz_cal = comm.bcast(dz_cal, root=0)

    if rank == 0:
        overseer_work(cube, wave, task_grain_size = 1, end=end, task_info = [path, filename, number])
    else:
        worker_work(rank, s1d, dz_cal)
        pass

    
# Remember to module load mpi/openmpi-401_gcc-485