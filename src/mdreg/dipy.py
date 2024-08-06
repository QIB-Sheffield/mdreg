import os
import multiprocessing
import numpy as np
from tqdm import tqdm
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric, EMMetric, SSDMetric

from mdreg import utils


def coreg_series(*args, parallel=False, **kwargs):
    if parallel:
        return _coreg_series_parallel(*args, **kwargs)
    else:
        return _coreg_series_sequential(*args, **kwargs)


def _coreg_series_sequential(moving:np.ndarray, fixed:np.ndarray, params):

    nt = moving.shape[-1]
    coreg, deformation = utils._init_output(moving)

    for t in tqdm(range(nt), desc='Coregistering series'): 
        coreg[...,t], deformation[...,t] = _coregister(
            moving[...,t], fixed[...,t], params)

    return coreg, deformation


def _coreg_series_parallel(moving:np.ndarray, fixed:np.ndarray, params):

    try: 
        num_workers = int(len(os.sched_getaffinity(0)))
    except: 
        num_workers = int(os.cpu_count())

    nt = moving.shape[-1]
    coreg, deformation = utils._init_output(moving)

    pool = multiprocessing.Pool(processes=num_workers)
    args = [(moving[...,t], fixed[...,t], params) for t in range(nt)]
    results = list(tqdm(pool.imap(_coregister_parallel, args), total=nt, desc='Coregistering series'))

    # Good practice to close and join when the pool is no longer needed
    # https://stackoverflow.com/questions/38271547/when-should-we-call-multiprocessing-pool-join
    pool.close()
    pool.join()

    for t in range(nt):
        coreg[...,t] = results[t][0]
        deformation[...,t] = results[t][1]

    return coreg, deformation
    

def _coregister_parallel(args):
    moving, fixed, parameters = args
    return _coregister(moving, fixed, parameters)


def _coregister(moving, fixed, parameters):
    
    dim = fixed.ndim

    # 3D registration does not seem to work with smaller slabs
    # Exclude this case
    if dim == 3:
        if fixed.shape[-1] < 6:
            msg = 'The 3D volume does not have enough slices for 3D registration. \n'
            msg += 'Try 2D registration instead.'
            raise ValueError(msg)
        
    # Define the metric
    metric = parameters['metric'] # default = metric="Cross-Correlation"
    if metric == "Cross-Correlation":
        sigma_diff = 3.0    # Gaussian Kernel
        radius = 4          # Window for local CC
        metric = CCMetric(dim, sigma_diff, radius)
    elif metric == 'Expectation-Maximization':
        metric = EMMetric(dim, smooth=1.0)
    elif metric == 'Sum of Squared Differences':
        metric = SSDMetric(dim, smooth=4.0)
    else:
        msg = 'The metric ' + metric + ' is currently not implemented.'
        raise ValueError(msg) 

    # Define the deformation model
    transformation = parameters['transform'] # default='Symmetric Diffeomorphic'
    if transformation == 'Symmetric Diffeomorphic':
        level_iters = [100, 50, 25]
        sdr = SymmetricDiffeomorphicRegistration(metric, level_iters, inv_iter=50)
    else:
        msg = 'The transform ' + transformation + ' is currently not implemented.'
        raise ValueError(msg) 

    # Perform the optimization, return a DiffeomorphicMap object
    mapping = sdr.optimize(fixed, moving)

    # Get forward deformation field
    deformation_field = mapping.get_forward_field()

    # Warp the moving image
    warped_moving = mapping.transform(moving, 'linear')

    return warped_moving.flatten(), deformation_field