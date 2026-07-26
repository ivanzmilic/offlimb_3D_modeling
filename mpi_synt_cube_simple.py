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
from scipy.interpolate import RegularGridInterpolator as rgi

# -----------------------------------------------------------------------------------------------------------------------------------

def airtovac(lambda_air):

    s = 1E4/(lambda_air*1E1);
    n = 1.0 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s*s) + 0.0001599740894897 / (38.92568793293 - s*s);
    return lambda_air * n

def synth(param_ray, boundary, wavelengths):
    
    # This function synthesises the spectrum for a given atmospheric column.
    #print('\n AAAAAAAAAAA \n',param_ray['Temperature'].shape)
    #exit();
    
    op, em = coe.calc_op_em(param_ray, wavelengths, take_given_S=False)
    spectrum_temp, tau, CR = coe.simple_formal_solution(op, op, 24.0e5)
    spectrum = 1.0 - np.exp(-tau)
    return spectrum

# But now we need a function that will create a ray from a given y,z slice, taking into account sphericity:

def create_curved_grid(z_index):
    
    # Meant for testing, but we can also just call it from create ray and then interpolate the slice to get the parameters at each point.
    N_steps = 10001
    
    delta_los = 24.0
    delta_z = 20.0  
    
    s = (np.arange(N_steps) - N_steps // 2) * delta_los # km    
    R_at_the_tangent = 696.0E3 + z_index * delta_z # km
    
    angle = np.arctan(s / R_at_the_tangent)
    y = R_at_the_tangent * angle
    z = z_index * delta_z + np.sqrt(s**2 + R_at_the_tangent**2) - R_at_the_tangent
    
    return s, y, z

def create_ray(slice, z_index):
    
    param_ray = 0
    
    # We need to create a grid of y, z values that traverses as it travels in the spherical geometry 
    # Then we bi-linearly interpolate where we can, and where we fall out - we just set zeros (or we don't even need anything)
    
    s,y,z = create_curved_grid(z_index)
    
    NY = slice['Temperature'].shape[0]
    NZ = slice['Temperature'].shape[1]
    
    delta_los = 24.0
    delta_z = 20.0
    
    # Then we just interpolate the slice to get the parameters at each point.
    y_slice = (np.arange(NY) - NY // 2) * delta_los
    z_slice = (np.arange(NZ)) * delta_z
    
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
        'Population_lower_level': PopL_interpolated/1E6, # We will choose to convert to cgs here
        'Population_upper_level': PopU_interpolated/1E6
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

    #print (task_end)
    
    sl = slice(task_start, task_end) # this is a slice object, allowing us to access the specific thingy
    
    #print (sl)
    data = {}
    data['taskGrainSize'] = task_end - task_start

    #print (data['z']/1E3)
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
        to_output.writeto('/dat/milic/offlimb_output.fits', overwrite=True)

    return 0  

def worker_work(rank):
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
                
                    I[i,j,:] = synth(param_ray, boundary=0, wavelengths=wave)
            
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
        number = 0
        
        #filename = sys.argv[2]  # characteristic naming for the files, used differently depending on what 
                                # kind of simulation we are working with 
        #number = int(sys.argv[3]) # number of the snapshot - again will be used differently for muram, co5bold, etc...
        #stokes = sys.argv[4].lower() == 'true' # whether to synthesize Stokes I or all 4 components
        #atmos_format = sys.argv[5]
        
        cube = 0
        
        cube = h5py.File(path,'r')

        wave = np.linspace(392.8,394.8,1001)
        
        
        print("info::overseer::input cube shape is: ", cube['Temperature'].shape)

        overseer_work(cube, wave, task_grain_size = 1, end=end, task_info = [path, filename, number])
    else:
        worker_work(rank)
        pass

    
# Remember to module load mpi/openmpi-401_gcc-485