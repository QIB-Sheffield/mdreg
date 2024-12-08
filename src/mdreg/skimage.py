import os
import multiprocessing
import numpy as np
from tqdm import tqdm

from skimage.registration import optical_flow_tvl1
from skimage.transform import warp as skiwarp

from mdreg import utils


def coreg(moving:np.ndarray, fixed:np.ndarray, **kwargs):

    """
    Coregister two arrays
    
    Parameters
    ----------
    moving : numpy.ndarray
        The moving image with dimensions (x,y) or (x,y,z). 
    fixed : numpy.ndarray
        The fixed target image with the same shape as the moving image. 
    kwargs : dict
        Any keyword argument accepted by `skimage.optical_flow_tvl1`. 
    
    Returns
    -------
    coreg : numpy.ndarray
        Coregistered image in the same shape as the moving image.
    deformation : numpy.ndarray
        Deformation field in the same shape as the moving image - but with an 
        additional dimension at the end for the components of the deformation 
        vector. 
    """

    if moving.ndim == 2: 
        return _coreg_2d(moving, fixed, **kwargs)
    if moving.ndim == 3:
        return _coreg_3d(moving, fixed, **kwargs)


def coreg_series(moving:np.ndarray, fixed:np.ndarray, parallel=False, 
                 progress_bar=False, **kwargs):
    """
    Coregister two series of 2D images or 3D volumes.

    Parameters
    ----------
    moving : numpy.ndarray
        The moving image or volume, with dimensions (x,y,t) or (x,y,z,t). 
    fixed : numpy.ndarray
        The fixed target image or volume, in the same dimensions as the 
        moving image. 
    parallel : bool
        Set to True to parallelize the computations. Defaults to False.
    progress_bar : bool
        Show a progress bar during the computation. This keyword is ignored 
        if parallel = True. Defaults to False.
    kwargs : dict
        Any keyword argument accepted by `skimage.optical_flow_tvl1`. 

    Returns
    -------
    coregistered : numpy.ndarray
        Coregistered series with the same dimensions as the moving image. 
    deformation : numpy.ndarray
        The deformation field with the same dimensions as *moving*, and one 
        additional dimension for the components of the vector field. If 
        *moving* 
        has dimensions (x,y,t) and (x,y,z,t), then the deformation field will 
        have dimensions (x,y,2,t) and (x,y,z,3,t), respectively.
    """

    if parallel:
        return _coreg_series_parallel(
            moving, fixed, **kwargs)
    else:
        return _coreg_series_sequential(
            moving, fixed, progress_bar=progress_bar, **kwargs)


def _coreg_series_sequential(moving, fixed, progress_bar=False, **kwargs):

    nt = moving.shape[-1]
    deformed, deformation = utils._init_output(moving)

    for t in tqdm(range(nt), 
                  desc='Coregistering series', 
                  disable=not progress_bar): 
        deformed[...,t], deformation[...,t] = coreg(
            moving[...,t], fixed[...,t], **kwargs)

    return deformed, deformation


def _coreg_series_parallel(moving, fixed, **kwargs):

    nt = moving.shape[-1]
    deformed, deformation = utils._init_output(moving)

    try: 
        num_workers = int(len(os.sched_getaffinity(0)))
    except: 
        num_workers = int(os.cpu_count())
        
    pool = multiprocessing.Pool(processes=num_workers)
    args = [(moving[...,t], fixed[...,t], kwargs) for t in range(nt)]
    results = list(tqdm(pool.imap(_coreg_parallel, args), total=nt, desc='Coregistering series'))

    # Good practice to close and join when the pool is no longer needed
    # https://stackoverflow.com/questions/38271547/when-should-we-call-multiprocessing-pool-join
    pool.close()
    pool.join()

    for t in range(nt):
        deformed[...,t] = results[t][0]
        deformation[...,t] = results[t][1]

    return deformed, deformation
    

def _coreg_parallel(args):
    moving, fixed, kwargs = args
    return coreg(moving, fixed, **kwargs)


def _coreg_2d(moving, fixed, **kwargs):

    # Does not work with float or mixed type for some reason
    moving, fixed, a, b, dtype = _torange(moving, fixed)
    
    rc, cc = np.meshgrid( 
        np.arange(moving.shape[0]), 
        np.arange(moving.shape[1]),
        indexing='ij')
    row_coords = rc
    col_coords = cc

    v, u = optical_flow_tvl1(fixed, moving, **kwargs)
    new_coords = np.array([row_coords + v, col_coords + u])
    deformation_field = np.stack([v, u], axis=-1)
    warped_moving = skiwarp(moving, new_coords, mode='edge', 
                            preserve_range=True)
    if a is not None:
        # Scale back to original range and type
        warped_moving = warped_moving.astype(dtype)
        warped_moving = (warped_moving-b)/a
        
    return warped_moving, deformation_field


def _coreg_3d(moving, fixed, **kwargs):

    moving, fixed, a, b, dtype = _torange(moving, fixed)
    
    rc, cc, sc = np.meshgrid( 
        np.arange(moving.shape[0]), 
        np.arange(moving.shape[1]),
        np.arange(moving.shape[2]),
        indexing='ij')
    row_coords = rc
    col_coords = cc
    slice_coords = sc

    v, u, w = optical_flow_tvl1(fixed, moving, **kwargs)
    new_coords = np.array([row_coords + v, col_coords + u, slice_coords+w])
    deformation_field = np.stack([v, u, w], axis=-1)
    warped_moving = skiwarp(moving, new_coords, mode='edge', 
                            preserve_range=True)

    if a is not None:
        # Scale back to original range and type
        warped_moving = warped_moving.astype(dtype)
        warped_moving = (warped_moving-b)/a

    return warped_moving, deformation_field


# Needs testing
def warp(moving, defo):

    if moving.ndim == 2:
        rc, cc = np.meshgrid( 
            np.arange(moving.shape[0]), 
            np.arange(moving.shape[1]),
            indexing='ij',
        )
        row_coords = rc
        col_coords = cc

        v = defo[:,:,:,0]
        u = defo[:,:,:,1]

        coords = np.array([row_coords + v, col_coords + u])
        return skiwarp(moving, coords, mode='edge', preserve_range=True)
    
    if moving.ndim == 3:
        rc, cc, sc = np.meshgrid( 
            np.arange(moving.shape[0]), 
            np.arange(moving.shape[1]),
            np.arange(moving.shape[2]),
            indexing='ij',
        )
        row_coords = rc
        col_coords = cc
        slice_coords = sc

        v = defo[:,:,:,0]
        u = defo[:,:,:,1]
        w = defo[:,:,:,2]

        coords = np.array([row_coords + v, col_coords + u, slice_coords+w])
        return skiwarp(moving, coords, mode='edge', preserve_range=True)


def _torange(moving, fixed):

    dtype = moving.dtype

    if dtype in [np.half, np.single, np.double, np.longdouble]:

        # Stay away from the boundaries
        i16 = np.iinfo(np.int16)
        imin = float(i16.min) + 16
        imax = float(i16.max) - 16

        # get scaling coefficients
        amin = np.amin([np.amin(moving), np.amin(fixed)])
        amax = np.amax([np.amax(moving), np.amax(fixed)])
        if amax == amin:
            a = 1
            b = - amin
        else:
            a = (imax-imin)/(amax-amin)
            b = - a * amin + imin

        # Scale to integer range
        moving = np.around(a*moving + b).astype(np.int16)
        fixed = np.around(a*fixed + b).astype(np.int16)

        return moving, fixed, a, b, dtype
    
    else:
    
        # Not clear why this is necessary but does not work otherwise
        moving = moving.astype(np.int16)
        fixed = fixed.astype(np.int16)

        return moving, fixed, None, None, None