import os
import multiprocessing
import numpy as np
from tqdm import tqdm

from skimage.registration import optical_flow_tvl1
from skimage.transform import warp

from mdreg import utils


def coreg_series(*args, parallel=False, **kwargs):

    if parallel:
        return _coreg_series_parallel(*args, **kwargs)
    else:
        return _coreg_series_sequential(*args, **kwargs)


def _coreg_series_sequential(
        moving:np.ndarray, 
        fixed:np.ndarray, 
        params=None):

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

    for t in tqdm(range(nt), desc='Coregistering series'): 

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


def coreg(moving, fixed, params=None, coords=None):

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
    warped_moving = warp(moving, new_coords, mode='edge', preserve_range=True)
    deformation_field = np.stack([v, u], axis=-1)
    return warped_moving, deformation_field


def params(**override):
    params = {
        'method': 'optical flow',
    }
    for key, val in override.items():
        params[key]=val
    return params

