import os
import multiprocessing
import numpy as np
from tqdm import tqdm

from skimage.registration import optical_flow_tvl1
from skimage.transform import warp

from mdreg import utils


def coreg_series(*args, parallel=False, **kwargs):
    
    """
    Coregister a series of images.

    Parameters
    ----------
    *args : dict
            Coregistration arguments.
    parallel : bool
            Whether to perform coregistration in parallel.
    **kwargs : dict
            Coregistration keyword arguments.

    Returns
    -------
    coreg : numpy.ndarray
            Coregistered series. The array can be either 3D or 4D with the 
            following shapes:
                - 3D: (X, Y, T)
                - 4D: (X, Y, Z, T)
            Here, X, Y, Z are the spatial dimensions and T is the dimension 
            denoting change e.g. temporal dimension or flip angle.
    deformation : numpy.ndarray
            Deformation field.
            The array can be either 4D or 5D with the following shapes:
                - 4D: (X, Y, 3, T)
                - 5D: (X, Y, Z, 3, T)
    """
    

    if parallel:
        return _coreg_series_parallel(*args, **kwargs)
    else:
        return _coreg_series_sequential(*args, **kwargs)


def _coreg_series_sequential(
        moving:np.ndarray, 
        fixed:np.ndarray, 
        params=None,
        coords=None,
        progress_bar=False):

    nt = moving.shape[-1]
    deformed, deformation = utils._init_output(moving)

    for t in tqdm(range(nt), desc='Coregistering series', disable=not progress_bar): 

        deformed[...,t], deformation[...,t] = coreg(
            moving[...,t], fixed[...,t], 
            params=params, coords=coords,
        )

    return deformed, deformation


def _coreg_series_parallel(moving:np.ndarray, fixed:np.ndarray, params=None):

    rc, cc = np.meshgrid( 
        np.arange(moving.shape[0]), 
        np.arange(moving.shape[1]),
        indexing='ij')
    coords = {
        'row_coords': rc,
        'col_coords': cc,
    }

    nt = moving.shape[-1]
    deformed, deformation = utils._init_output(moving)

    try: 
        num_workers = int(len(os.sched_getaffinity(0)))
    except: 
        num_workers = int(os.cpu_count())
        
    pool = multiprocessing.Pool(processes=num_workers)
    args = [(moving[...,t], fixed[...,t], params, coords) for t in range(nt)]
    results = list(tqdm(pool.imap(_coregister_parallel, args), total=nt, desc='Coregistering series'))

    # Good practice to close and join when the pool is no longer needed
    # https://stackoverflow.com/questions/38271547/when-should-we-call-multiprocessing-pool-join
    pool.close()
    pool.join()

    for t in range(nt):
        deformed[...,t] = results[t][0]
        deformation[...,t] = results[t][1]

    return deformed, deformation
    

def _coregister_parallel(args):
    moving, fixed, params, coords = args
    return coreg(moving, fixed, params=params, coords=coords)

def coreg(source:np.ndarray, *args, **kwargs):

    """
    Coregister two arrays
    
    Parameters
    ----------
    source : numpy.ndarray
        The source image.
    *args : dict
        Coregistration arguments.
    **kwargs : dict
        Coregistration keyword arguments.
    
    Returns
    -------
    coreg : numpy.ndarray
        Coregistered image.
    deformation : numpy.ndarray
        Deformation field.
    
    """

    if source.ndim == 2: 
        return _coreg_2d(source, *args, **kwargs)
    
    if source.ndim == 3:
        return _coreg_3d(source, *args, **kwargs)


def _coreg_2d(moving, fixed, params=None, coords=None):

    # Does not work with float or mixed type for some reason
    moving = moving.astype(np.int16)
    fixed = fixed.astype(np.int16)
    
    if params is None:
        params = {'method': 'optical flow'}
    if 'method' not in params:
        params['method'] = 'optical flow'
    
    if coords is None:
        rc, cc = np.meshgrid( 
            np.arange(moving.shape[0]), 
            np.arange(moving.shape[1]),
            indexing='ij')
        row_coords = rc
        col_coords = cc
    else:
        row_coords = coords['row_coords']
        col_coords = coords['col_coords']

    if params['method'] == 'optical flow':
        kwargs = {i:params[i] for i in params if i!='method'}
        v, u = optical_flow_tvl1(fixed, moving, **kwargs)
    else:
        raise ValueError('This method is not currently available')
    
    new_coords = np.array([row_coords + v, col_coords + u])
    deformation_field = np.stack([v, u], axis=-1)

    warped_moving = warp(moving, new_coords, mode='edge', preserve_range=True)
    return warped_moving, deformation_field

def _coreg_3d(moving, fixed, params=None, coords=None):

    # Does not work with float or mixed type for some reason
    moving = moving.astype(np.int16)
    fixed = fixed.astype(np.int16)
    
    if params is None:
        params = {'method': 'optical flow'}
    if 'method' not in params:
        params['method'] = 'optical flow'
    
    if coords is None:
        rc, cc, sc = np.meshgrid( 
            np.arange(moving.shape[0]), 
            np.arange(moving.shape[1]),
            np.arange(moving.shape[2]),
            indexing='ij')
        row_coords = rc
        col_coords = cc
        slice_coords = sc
    else:
        row_coords = coords['row_coords']
        col_coords = coords['col_coords']
        slice_coords = coords['slice_coords']

    if params['method'] == 'optical flow':
        kwargs = {i:params[i] for i in params if i!='method'}
        v, u, w = optical_flow_tvl1(fixed, moving, **kwargs)
    else:
        raise ValueError('This method is not currently available')
    
    new_coords = np.array([row_coords + v, col_coords + u, slice_coords+w])
    deformation_field = np.stack([v, u, w], axis=-1)

    warped_moving = warp(moving, new_coords, mode='edge', preserve_range=True)
    return warped_moving, deformation_field


def params(**override):

    """
    Generate parameters for skimage registration.

    Parameters
    ----------
    default : str
        The default parameter set to use.
    **override : dict
        Parameters to override.
    
    Returns
    -------
    params : dict
        The parameter set.

    """

    params = {
        'method': 'optical flow',
    }
    for key, val in override.items():
        params[key]=val
    return params

