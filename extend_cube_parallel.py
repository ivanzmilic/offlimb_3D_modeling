import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import muram as mio
import sys
import time
import mmap
import h5py
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


# ======================================================================================
# Faster / parallel version of extend_cube.py, with two run modes.
#
# Shared optimization (both modes): the (x,y) query points are identical for every
# z-layer k AND every physical quantity, so we precompute the bilinear-interpolation
# flat indices + weights ONCE. Each slice interpolation is then a 4-point gather +
# weighted sum (no per-slice RegularGridInterpolator, no inner loop over LOS). This
# turns ~2.46M interpolator calls into a handful of vectorized gathers.
#
# MODE = 'ram'    : loop parallel over z-layers; workers write into one anonymous
#                   shared-memory buffer (fork-inherited). Fastest, needs the whole
#                   output in RAM (~123 GB float32 / ~246 GB float64).
#
# MODE = 'stream' : loop parallel over LOS-blocks; each worker returns one contiguous
#                   [i0:i1, :, :] block, the parent (single HDF5 writer) streams it
#                   straight to disk and discards it. RAM stays at a few GB. The block
#                   is written contiguously and aligned with the synthesis read grain
#                   (cube['Temperature'][task_start:task_end, :, :]).
#
# CLI:
#   python extend_cube_parallel.py <muram_cube_dir> <snapno> <lw_synth.fits> \
#          <z_offset> <output.h5> [nproc] [mode=ram|stream] [los_block]
# ======================================================================================

# --- knobs -----------------------------------------------------------------------------
# Output dtype. MURaM data is float32 natively, so float32 halves RAM, disk, and memory
# bandwidth with no meaningful precision loss. Set to np.float64 for bit-identical output
# to the original script.
OUT_DTYPE = np.float32

# Geometry / grid (hardcoded, matching extend_cube.py; TODOs carried over):
ANGLE_DEG = 80          # TODO: hardcoded
DELTA_X = 24            # km # TODO: should be taken from the header (true dX ~ 23.4375 km)
DELTA_Y = 24            # km
LINE_LENGTH_KM = 3e5    # TODO: hardcoded

QUANTITY_KEYS = ["Temperature", "Pressure", "Electron_density", "LOS_velocity",
                 "Population_lower_level", "Population_upper_level"]
# ---------------------------------------------------------------------------------------

# Globals populated in __main__ before the worker pool is forked, so workers inherit them.
cube_new = None          # 'ram' mode only: shared-memory output
_snap_slices = None      # (Temp, Pres, ne, vy, vz) MURaM memmaps
pops = None
z_offset = 0
n_los = 0
n_steps = 0
Nz = 0
# precomputed bilinear gather (flat indices into a C-order (Nx,Ny) slice) + weights:
_i00 = _i10 = _i01 = _i11 = None
_w00 = _w10 = _w01 = _w11 = None


def _interp_full(slice2d):
    """Bilinear-interpolate a raw 2D (Nx, Ny) field onto the whole folded LOS grid."""
    f = np.ascontiguousarray(slice2d).reshape(-1)
    v = (f[_i00] * _w00 + f[_i10] * _w10 + f[_i01] * _w01 + f[_i11] * _w11)
    return v.reshape(n_los, n_steps)


def _process_k_list(k_list):
    """'ram' mode worker: fill cube_new[:, :, :, k] for every k in k_list."""
    cos_a = np.cos(np.radians(ANGLE_DEG))
    sin_a = np.sin(np.radians(ANGLE_DEG))
    Temp, Pres, ne, vy, vz = _snap_slices
    for k in k_list:
        km = k + z_offset                                  # z-aligned MURaM layer (the fix)
        cube_new[0, :, :, k] = _interp_full(Temp[km, :, :])
        cube_new[1, :, :, k] = _interp_full(Pres[km, :, :])
        cube_new[2, :, :, k] = _interp_full(ne[km, :, :])
        cube_new[3, :, :, k] = _interp_full(vy[km, :, :] * cos_a + vz[km, :, :] * sin_a)
        # Level populations: NO z_offset (they already span the cropped range).
        cube_new[4, :, :, k] = _interp_full(pops[:, :, 0, k])
        cube_new[5, :, :, k] = _interp_full(pops[:, :, 1, k])
    return len(k_list)


def _process_los_block(bounds):
    """'stream' mode worker: return one contiguous (6, B, n_steps, Nz) LOS block.

    The population z-flip is applied here (written to kf = Nz-1-k), so the parent just
    writes the block out verbatim.
    """
    i0, i1 = bounds
    B = i1 - i0
    s = slice(i0 * n_steps, i1 * n_steps)                  # this block's flat rows
    i00, i10, i01, i11 = _i00[s], _i10[s], _i01[s], _i11[s]
    w00, w10, w01, w11 = _w00[s], _w10[s], _w01[s], _w11[s]

    def interp(slice2d):
        f = np.ascontiguousarray(slice2d).reshape(-1)
        v = f[i00] * w00 + f[i10] * w10 + f[i01] * w01 + f[i11] * w11
        return v.reshape(B, n_steps)

    Temp, Pres, ne, vy, vz = _snap_slices
    cos_a = np.cos(np.radians(ANGLE_DEG))
    sin_a = np.sin(np.radians(ANGLE_DEG))
    buf = np.empty((6, B, n_steps, Nz), dtype=OUT_DTYPE)
    for k in range(Nz):
        km = k + z_offset
        buf[0, :, :, k] = interp(Temp[km, :, :])
        buf[1, :, :, k] = interp(Pres[km, :, :])
        buf[2, :, :, k] = interp(ne[km, :, :])
        buf[3, :, :, k] = interp(vy[km, :, :] * cos_a + vz[km, :, :] * sin_a)
        kf = Nz - 1 - k                                    # population flip, on the fly
        buf[4, :, :, kf] = interp(pops[:, :, 0, k])
        buf[5, :, :, kf] = interp(pops[:, :, 1, k])
    return bounds, buf


if __name__ == '__main__':

    muram_cube = sys.argv[1]
    snapno = int(sys.argv[2])
    lw_synth = sys.argv[3]
    z_offset = int(sys.argv[4])
    final_output_file = sys.argv[5]
    nproc = int(sys.argv[6]) if len(sys.argv) > 6 else min(32, mp.cpu_count())
    mode = sys.argv[7] if len(sys.argv) > 7 else 'ram'
    los_block = int(sys.argv[8]) if len(sys.argv) > 8 else 16   # 'stream' block width
    assert mode in ('ram', 'stream'), "mode must be 'ram' or 'stream'"

    t0 = time.time()

    # --- inputs ------------------------------------------------------------------------
    muram_snap = mio.MuramSnap(muram_cube, snapno)
    print(muram_snap.available)
    print("info::shape of the original cube: ", muram_snap.Temp.shape)

    lw_output = fits.open(lw_synth)
    lw_output.info()
    pops = lw_output[2].data
    print("info::pops shape: ", pops.shape)

    tau_mean = np.mean(muram_snap.tau, axis=(1, 2))
    z_idx_ph = np.argmin(np.abs(tau_mean - 1))
    print("info:: the index of the tau=1 layer is: ", z_idx_ph,
          "and tau there is = ", tau_mean[z_idx_ph])

    Nx = pops.shape[0]
    Ny = pops.shape[1]
    Nz = pops.shape[3]
    n_los = Nx
    n_steps = int(LINE_LENGTH_KM / DELTA_X)
    print("number of steps is: ", n_steps)

    assert Nz + z_offset <= muram_snap.Temp.shape[0], \
        "z_offset pushes the population range past the top of the MURaM cube"

    # --- folded LOS query coordinates (identical for all k, all quantities) ------------
    angle_rad = np.radians(ANGLE_DEG)
    steps = (np.arange(n_steps) + 0.5)
    i_starts = np.arange(n_los) * DELTA_X
    x_coords_grid = i_starts[:, None] + steps[None, :] * DELTA_X * np.cos(angle_rad)
    y_coords_grid = np.broadcast_to(steps[None, :] * DELTA_Y * np.sin(angle_rad),
                                    (n_los, n_steps)).copy()
    x_coords_grid_in_box = x_coords_grid % ((Nx - 1) * DELTA_X)
    y_coords_grid_in_box = y_coords_grid % ((Ny - 1) * DELTA_Y)
    print("info::these are the y coordinates we are interpolating at: ",
          y_coords_grid_in_box[0])

    # --- precompute bilinear indices + weights ONCE ------------------------------------
    xq = x_coords_grid_in_box.ravel()
    yq = y_coords_grid_in_box.ravel()
    fx = xq / DELTA_X
    fy = yq / DELTA_Y
    ix0 = np.floor(fx).astype(np.int64)
    iy0 = np.floor(fy).astype(np.int64)
    wx = (fx - ix0).astype(np.float32)
    wy = (fy - iy0).astype(np.float32)
    ix0 = np.clip(ix0, 0, Nx - 1); ix1 = np.clip(ix0 + 1, 0, Nx - 1)
    iy0 = np.clip(iy0, 0, Ny - 1); iy1 = np.clip(iy0 + 1, 0, Ny - 1)
    _i00 = (ix0 * Ny + iy0).astype(np.int32)
    _i10 = (ix1 * Ny + iy0).astype(np.int32)
    _i01 = (ix0 * Ny + iy1).astype(np.int32)
    _i11 = (ix1 * Ny + iy1).astype(np.int32)
    _w00 = (1 - wx) * (1 - wy)
    _w10 = wx * (1 - wy)
    _w01 = (1 - wx) * wy
    _w11 = wx * wy
    del xq, yq, fx, fy, ix0, ix1, iy0, iy1, wx, wy

    # MURaM memmaps materialized in the parent so workers inherit them pre-fork:
    _snap_slices = (muram_snap.Temp, muram_snap.Pres, muram_snap.ne,
                    muram_snap.vy, muram_snap.vz)
    fork_ctx = mp.get_context('fork')

    aux = {
        "x_coords_grid_in_box": x_coords_grid_in_box,
        "y_coords_grid_in_box": y_coords_grid_in_box,
        "x_coords_grid": x_coords_grid,
        "y_coords_grid": y_coords_grid,
    }

    if mode == 'ram':
        # ------------------------------------------------------------------ ram mode ---
        shape = (6, n_los, n_steps, Nz)
        nbytes = int(np.prod(shape)) * np.dtype(OUT_DTYPE).itemsize
        print("info::[ram] allocating shared output %s = %.1f GB (%s)"
              % (shape, nbytes / 1e9, np.dtype(OUT_DTYPE).name))
        _shared_buf = mmap.mmap(-1, nbytes, flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS)
        cube_new = np.frombuffer(_shared_buf, dtype=OUT_DTYPE).reshape(shape)

        k_chunks = [c for c in np.array_split(np.arange(Nz), max(nproc, 1)) if len(c)]
        print("info::[ram] filling %d layers with %d worker(s)..." % (Nz, nproc))
        if nproc <= 1:
            for c in k_chunks:
                _process_k_list(c)
        else:
            with ProcessPoolExecutor(max_workers=nproc, mp_context=fork_ctx) as ex:
                done = 0
                for got in ex.map(_process_k_list, k_chunks):
                    done += got
                    print("info::  ...%d/%d layers done" % (done, Nz))
        print("info::[ram] fill done in %.1f s" % (time.time() - t0))

        # population z-flip (explicit copy so the reversed view can't alias unsafely):
        cube_new[4] = cube_new[4, :, :, ::-1].copy()
        cube_new[5] = cube_new[5, :, :, ::-1].copy()

        plt.figure(figsize=[14, 6])
        plt.imshow(cube_new[0, :, :, 55], origin="lower", cmap='inferno',
                   aspect='equal', rasterized=True)
        plt.colorbar()
        plt.savefig("temperature_extension_sanity_check.png", dpi=300, bbox_inches='tight')

        print("info::[ram] writing %s ..." % final_output_file)
        with h5py.File(final_output_file, 'w') as f:
            for q, key in enumerate(QUANTITY_KEYS):
                f.create_dataset(key, data=cube_new[q])
            for key, value in aux.items():
                f.create_dataset(key, data=value)

    else:
        # --------------------------------------------------------------- stream mode ---
        blocks = [(i0, min(i0 + los_block, n_los)) for i0 in range(0, n_los, los_block)]
        block_bytes = 6 * los_block * n_steps * Nz * np.dtype(OUT_DTYPE).itemsize
        window = max(nproc + 2, 2) if nproc > 1 else 1
        print("info::[stream] %d LOS-blocks of width %d; ~%.2f GB/block; "
              "RAM ceiling ~%.1f GB (window=%d)"
              % (len(blocks), los_block, block_bytes / 1e9,
                 block_bytes * window / 1e9, window))

        with h5py.File(final_output_file, 'w') as f:
            dsets = {key: f.create_dataset(key, shape=(n_los, n_steps, Nz),
                                           dtype=OUT_DTYPE)
                     for key in QUANTITY_KEYS}

            def _write(block_bounds, buf):
                i0, i1 = block_bounds
                for q, key in enumerate(QUANTITY_KEYS):
                    dsets[key][i0:i1, :, :] = buf[q]

            done = 0
            if nproc <= 1:
                for b in blocks:
                    _, buf = _process_los_block(b)
                    _write(b, buf)
                    done += 1
                    print("info::  ...%d/%d blocks done" % (done, len(blocks)))
            else:
                # Bounded sliding window: at most `window` blocks in flight, so RAM
                # stays ~window*block_bytes regardless of how far compute runs ahead.
                with ProcessPoolExecutor(max_workers=nproc, mp_context=fork_ctx) as ex:
                    it = iter(blocks)
                    inflight = deque()
                    for _ in range(window):
                        b = next(it, None)
                        if b is None:
                            break
                        inflight.append(ex.submit(_process_los_block, b))
                    while inflight:
                        b_bounds, buf = inflight.popleft().result()
                        _write(b_bounds, buf)
                        done += 1
                        print("info::  ...%d/%d blocks done" % (done, len(blocks)))
                        b = next(it, None)
                        if b is not None:
                            inflight.append(ex.submit(_process_los_block, b))

            for key, value in aux.items():
                f.create_dataset(key, data=value)

            # sanity plot: read one z-layer back from disk (one-time strided read)
            t_layer = f["Temperature"][:, :, 55]
        plt.figure(figsize=[14, 6])
        plt.imshow(t_layer, origin="lower", cmap='inferno', aspect='equal', rasterized=True)
        plt.colorbar()
        plt.savefig("temperature_extension_sanity_check.png", dpi=300, bbox_inches='tight')

    print("info::all done in %.1f s (mode=%s)" % (time.time() - t0, mode))
