import numpy as np
import os
import h5py
import hdf5plugin
import dask.array as da
import psutil
from dask.diagnostics import ProgressBar

def read_empad(fname):
    """
    Reads the EMPAD file at filename, returning a 4D numpy array .

    EMPAD files are 130x128 arrays, consisting of 128x128 arrays of data followed by
    two rows of metadata.  The metadata holds the scan size.
    The function determines the scan size by extracting the first and last frames' scan position.
    Then the data set is shaped.

    Arguments:
        fname     path to the EMPAD file

    Returns:
        data      datacube, excluding the metadata rows.
    """
    print("Reading EMPAD raw data file "+fname)
    rows = 130
    cols = 128
    filesize = os.path.getsize(fname)
    framesize = rows * cols * 4  # 4 bytes per pixel
    NFrames = filesize / framesize
    data_shape = (int(NFrames), rows, cols)
    with open(fname, "rb") as fid:
        data = np.fromfile(fid, np.float32).reshape(data_shape)[:, :rows, :]
    # Get the scan shape
    (dx,dy)=(1+data[-1,129:130,12]-data[0,129:130,12],1+data[-1,129:130,13]-data[0,129:130,13])
    data_shape=(int(dx.item()),int(dy.item()),rows,cols)  
    data=data.reshape(data_shape)
    return data[:,:,0:128,:]


def _h5_group_to_dict(h5_obj):
    """Recursively converts HDF5 attributes and scalar datasets into a Python dict."""
    out = {}
    
    # Extract HDF5 attributes on the current group/dataset
    for key, val in h5_obj.attrs.items():
        if isinstance(val, np.bytes_):
            val = val.decode('utf-8', errors='ignore')
        elif isinstance(val, np.ndarray) and val.dtype.kind == 'S':
            val = [x.decode('utf-8', errors='ignore') for x in val]
        out[f"attr_{key}"] = val

    # Traverse child groups and non-large datasets
    if isinstance(h5_obj, h5py.Group):
        for key, item in h5_obj.items():
            # Skip massive multidimensional detector data streams
            if key.startswith('data_') and isinstance(item, h5py.Dataset) and item.ndim >= 3:
                continue
            
            if isinstance(item, h5py.Group):
                out[key] = _h5_group_to_dict(item)
            elif isinstance(item, h5py.Dataset):
                # Only read small datasets/scalars into metadata
                if item.size <= 100:
                    val = item[()]
                    if isinstance(val, np.bytes_):
                        val = val.decode('utf-8', errors='ignore')
                    elif isinstance(val, np.ndarray) and val.dtype.kind == 'S':
                        val = [x.decode('utf-8', errors='ignore') for x in val]
                    elif isinstance(val, np.ndarray):
                        val = val.tolist()
                    out[key] = val
                else:
                    out[key] = f"<Dataset shape={item.shape} dtype={item.dtype}>"
    return out


def load_dectris_em_metadata(em_metadata_filepath):
    with h5py.File(em_metadata_filepath, 'r') as h5file:
        data_grp = h5file.get('/electron_microscope/scan_controller/regular_scan')
        
        if data_grp is None or not list(data_grp.keys()):
            raise KeyError("No datasets found under '/electron_microscope/scan_controller/regular_scan'.")
        
        # Extract scalar values with [()] before the file closes
        nx = h5file['/electron_microscope/scan_controller/regular_scan/n_pixels_x'][()]
        ny = h5file['/electron_microscope/scan_controller/regular_scan/n_pixels_y'][()]
        
        # Convert HDF5/NumPy scalar types to standard Python ints
        scan_shape = (int(nx), int(ny))

        # Extract scalar values with [()] before the file closes
        magnification = float(h5file['/electron_microscope/illumination_system/scan_magnification'][()])
        # Extract scalar values with [()] before the file closes
        camera_length = float(h5file['/electron_microscope/imaging_system/camera_length'][()])
        # Convert HDF5/NumPy scalar types to standard Python ints
        
        # Recursively parse metadata into memory
        em_meta_dict = _h5_group_to_dict(h5file)
        
    return scan_shape, magnification, camera_length, em_meta_dict

def load_dectris_dask_binned(
    master_filepath, 
    scan_shape=None, 
    bin_scan=(1, 1), 
    bin_det=(1, 1),
    chunk_frames=2048, 
    reduction='sum',
    max_ram_fraction=0.8,
    **kwargs):
    """
    Loads DECTRIS Eiger 4D STEM data with optional scan/detector binning, 
    a terminal progress bar, and memory safety checks.
    
    Any additional **kwargs are passed directly to `h5py.File()` (e.g. driver, rdcc_nbytes).
    """
    bin_sy, bin_sx = bin_scan
    bin_dy, bin_dx = bin_det

    with h5py.File(master_filepath, 'r', **kwargs) as f:
        # --- 1. RECURSIVELY EXTRACT MASTER METADATA ---
        master_metadata = _h5_group_to_dict(f)

        data_grp = f['/entry/data']
        data_keys = sorted([k for k in data_grp.keys() if k.startswith('data_')])
        
        if not data_keys:
            raise KeyError("No sub-datasets starting with 'data_' found under '/entry/data'.")

        # Automatically determine scan shape if not provided
        if scan_shape is None:
            scan_y_node = f.get('/entry/instrument/scan/Ny') or f.get('/entry/instrument/NDArray/Ny')
            scan_x_node = f.get('/entry/instrument/scan/Nx') or f.get('/entry/instrument/NDArray/Nx')

            if scan_y_node is not None and scan_x_node is not None:
                scan_y = int(scan_y_node[()])
                scan_x = int(scan_x_node[()])
            elif '/entry/instrument/detector/ntrigger' in f:
                n_frames = int(f['/entry/instrument/detector/ntrigger'][()])
                side = int(np.sqrt(n_frames))
                if side * side == n_frames:
                    scan_y, scan_x = side, side
                else:
                    raise ValueError(f"Cannot infer square scan shape from total frame count ({n_frames}). Pass `scan_shape=(Ny, Nx)`.")
            else:
                raise KeyError("Could not automatically infer scan shape. Please pass `scan_shape=(Ny, Nx)`.")
        else:
            scan_y, scan_x = scan_shape

        if scan_y % bin_sy != 0 or scan_x % bin_sx != 0:
            raise ValueError(f"Scan shape ({scan_y}, {scan_x}) must be divisible by bin_scan {bin_scan}.")

        # --- 2. BUILD LAZY DASK ARRAY ---
        first_dset = data_grp[data_keys[0]]
        det_y, det_x = first_dset.shape[1], first_dset.shape[2]

        if det_y % bin_dy != 0 or det_x % bin_dx != 0:
            raise ValueError(f"Detector frame shape ({det_y}, {det_x}) must be divisible by bin_det {bin_det}.")

        sub_arrays = []
        for key in data_keys:
            dset = data_grp[key]
            chunks = (min(chunk_frames, dset.shape[0]), det_y, det_x)
            sub_arrays.append(da.from_array(dset, chunks=chunks))

        all_frames = da.concatenate(sub_arrays, axis=0)
        
        # Reshape to unbinned 4D STEM layout
        total_frames = scan_y * scan_x
        dask_4d = all_frames[:total_frames].reshape((scan_y, scan_x, det_y, det_x))

        # --- 3. MULTI-DIMENSIONAL BINNING ---
        coarsen_axes = {}
        if bin_sy > 1: coarsen_axes[0] = bin_sy
        if bin_sx > 1: coarsen_axes[1] = bin_sx
        if bin_dy > 1: coarsen_axes[2] = bin_dy
        if bin_dx > 1: coarsen_axes[3] = bin_dx

        reduce_func = np.sum if reduction == 'sum' else np.mean
        if coarsen_axes:
            dask_binned = da.coarsen(
                reduce_func, 
                dask_4d, 
                axes=coarsen_axes, 
                trim_excess=True
            )
        else:
            dask_binned = dask_4d

        # --- 4. MEMORY CHECK ---
        required_bytes = dask_binned.nbytes
        available_bytes = psutil.virtual_memory().available
        allowed_bytes = available_bytes * max_ram_fraction

        req_gb = required_bytes / (1024**3)
        unbinned_gb = dask_4d.nbytes / (1024**3)
        avail_gb = available_bytes / (1024**3)

        print(f"Original 4D Shape: {dask_4d.shape} ({unbinned_gb:.2f} GB)")
        print(f"Binned 4D Shape:   {dask_binned.shape} ({req_gb:.2f} GB)")
        print(f"Available RAM:     {avail_gb:.2f} GB")

        if required_bytes > allowed_bytes:
            raise MemoryError(
                f"Dataset requires {req_gb:.2f} GB, exceeding {max_ram_fraction*100:.0f}% "
                f"of available RAM ({avail_gb:.2f} GB)."
            )

        # --- 5. COMPUTE WITH PROGRESS BAR ---
        print("Decompressing and computing binned 4D array...")
        with ProgressBar():
            data_4d = dask_binned.compute()

    return data_4d, master_metadata
