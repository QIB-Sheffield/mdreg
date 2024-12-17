import __main__
import os
import warnings
import multiprocessing
import numpy as np
from tqdm import tqdm
import itk
from skimage.measure import block_reduce
from mdreg import utils



def coreg(moving:np.ndarray, fixed:np.ndarray, spacing=1.0, downsample=1, 
          log=False, method='bspline', **params):

    """
    Coregister two arrays
    
    Parameters
    ----------
    moving : numpy.ndarray
        The moving image with dimensions (x,y) or (x,y,z). 
    fixed : numpy.ndarray
        The fixed target image with the same shape as the moving image. 
    spacing: array-like
        Pixel spacing in mm. This can be a single scalar if all dimensions 
        are equal, or an array of 2 elements (for 2D data) or 3 elements (
        for 3D data). Defaults to 1. 
    downsample: float
        Speed up the registration by downsampling the images with this factor. 
        downsample=1 means no downsampling. This is the default.
    log: bool
        Print a log during computations. Defaults to False.
    method : str
        Deformation method to use. Options are 'bspline', 'affine', 'rigid' 
        or 'translation'. Default is 'bspline'.
    params : dict
        Use keyword arguments to overrule any of the default parameters in 
        the elastix template for the chosen method.  
    
    Returns
    -------
    coreg : numpy.ndarray
        Coregistered image in the same shape as the moving image.
    deformation : numpy.ndarray
        Deformation field in the same shape as the moving image - but with an 
        additional dimension at the end for the components of the deformation 
        vector. 
    
    """

    # Masking is not yet implemented
    mask = None

    params_obj = _params_obj(method, **params) 

    if moving.ndim == 2: 
        coreg, defo = _coreg_2d(moving, fixed, spacing, downsample, 
                         log, mask, params_obj)
    
    if moving.ndim == 3:
        coreg, defo = _coreg_3d(moving, fixed, spacing, downsample, 
                         log, mask, params_obj)
        
    _cleanup(**params)
        
    return coreg, defo


def coreg_series(
        moving:np.ndarray, fixed:np.ndarray, spacing=1.0, downsample=1, 
        log=False, progress_bar=False, method='bspline', **params):
    
    """
    Coregister two series of 2D images or 3D volumes.

    Parameters
    ----------
    moving : numpy.ndarray
        The moving image or volume, with dimensions (x,y,t) or (x,y,z,t). 
    fixed : numpy.ndarray
        The fixed target image or volume, in the same dimensions as the 
        moving image. 
    spacing: array-like
        Pixel spacing in mm. This can be a single scalar if all dimensions 
        are equal, or an array of 2 elements (for 2D data) or 3 elements (
        for 3D data). Defaults to 1. 
    downsample: float
        Speed up the registration by downsampling the images with this factor. 
        downsample=1 means no downsampling. This is the default.
    log: bool
        Print a log during computations. Defaults to False.
    progress_bar: bool
        Display a progress bar during coregistration. Defaults to False.
    method : str
        Deformation method to use. Options are 'bspline', 'affine', 'rigid' 
        or 'translation'. Default is 'bspline'.
    params : dict
        Use keyword arguments to overrule any of the default parameters in 
        the elastix template for the chosen method.  

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

    if np.shape(moving) != np.shape(fixed):
        raise ValueError('Moving and fixed arrays must have the '
                         'same dimensions.')
    if np.sum(np.isnan(moving)) > 0:
        raise ValueError('Source image contains NaN values - cannot '
                         'perform coregistration')
    if np.sum(np.isnan(fixed)) > 0:
        raise ValueError('Target image contains NaN values - cannot '
                         'perform coregistration')
    
    # There is currently no benefit in parallellization with elastix 
    # because creating of the parameter object takes a long time and 
    # must be done for each pairwise coregistration - removing all 
    # potential benefit of the parallellization. Once the issues
    # has been resolved, this option can be offered as a keyword argument. 
    parallel = False

    # Masking is not yet implemented
    mask = None

    if parallel:
        coreg, defo = _coreg_series_parallel(
            moving, fixed, spacing, downsample, log, mask, 
            method, **params)
    else:
        coreg, defo = _coreg_series_sequential(
            moving, fixed, spacing, downsample, log, mask, 
            progress_bar, method, **params)

    _cleanup(**params)

    return coreg, defo
    

def _coreg_series_sequential(moving, fixed, spacing, downsample, log, mask, 
                             progress_bar, method, **params):

    # This is a very slow step so needs to be done outside the loop
    p_obj = _params_obj(method, **params) 

    deformed, deformation = utils._init_output(moving)
    for t in tqdm(range(moving.shape[-1]), 
                  desc='Coregistering series', 
                  disable=not progress_bar): 

        if mask is not None:
            mask_t = mask[...,t]
        else: 
            mask_t = None

        deformed[...,t], deformation[...,t] = _coreg(
            moving[...,t], fixed[...,t],
            spacing, downsample, log, mask_t, p_obj)
        
    return deformed, deformation


def _coreg_series_parallel(moving, fixed, spacing, downsample, log, mask, 
                           method, **params):

    # itk.force_load() # should speed up (but doesn't)
    # https://github.com/InsightSoftwareConsortium/ITKElastix/issues/204

    try: 
        num_workers = int(len(os.sched_getaffinity(0)))
    except: 
        num_workers = int(os.cpu_count())

    # Build list of arguments
    args = []
    for t in range(moving.shape[-1]):
        if mask is not None:
            mask_t = mask[...,t]
        else: 
            mask_t = None
        args_t = (moving[...,t], fixed[...,t], spacing, 
                  downsample, log, mask_t, method, params)
        args.append(args_t)

    # Process list of arguments in parallel
    pool = multiprocessing.Pool(processes=num_workers)
    results = pool.map(_coreg_parallel, args)
    # Good practice to close and join when the pool is no longer needed
    # https://stackoverflow.com/questions/38271547/when-should-we-call-multiprocessing-pool-join
    pool.close()
    pool.join()

    # Reformat list of results into arrays
    coreg, deformation = utils._init_output(moving)
    for t in range(moving.shape[-1]):
        coreg[...,t] = results[t][0]
        deformation[...,t] = results[t][1]

    return coreg, deformation


def _coreg_parallel(args):
    moving, fixed, spacing, downsample, log, mask, method, params = args
    p_obj = _params_obj(method, **params) 
    return _coreg(moving, fixed, spacing, downsample, log, mask, p_obj)


def _params_obj(method, **params):
    param_obj = itk.ParameterObject.New() # long runtime ~20s
    parameter_map_bspline = param_obj.GetDefaultParameterMap(method) 
    param_obj.AddParameterMap(parameter_map_bspline)
    for key, val in params.items():
        param_obj.SetParameter(key, str(val))
    return param_obj


def _coreg(moving, *args):
    if moving.ndim == 2: 
        return _coreg_2d(moving, *args)
    if moving.ndim == 3:
        return _coreg_3d(moving, *args)



def _coreg_2d(source_large, target_large, spacing, downsample, log, mask, 
              params_obj):

    if np.isscalar(spacing):
        spacing = [spacing, spacing]

    # Downsample source and target
    # The origin of an image is the center of the voxel in the lower left corner
    # The origin of the large image is (0,0).
    # The original of the small image is therefore: 
    #   spacing_large/2 + (spacing_small/2 - spacing_large)
    #   = (spacing_small - spacing_large)/2
    target_small = block_reduce(target_large, block_size=downsample, func=np.mean)
    source_small = block_reduce(source_large, block_size=downsample, func=np.mean)

    spacing_large = [spacing[1], spacing[0]] # correct numpy(x,y) ordering for itk(y,x)
    spacing_small = [spacing_large[0]*downsample, spacing_large[1]*downsample] # downsample large spacing

    spacing_large_y, spacing_large_x = spacing_large # seperate for straightforward readable implementation with ITK ordering
    spacing_small_y, spacing_small_x = spacing_small

    origin_large = [0,0]
    origin_small = [(spacing_small_y - spacing_large_y)/2, (spacing_small_x - spacing_large_x) / 2]

    # Coregister downsampled source to target
    source_small = np.ascontiguousarray(source_small.astype(np.float32))
    target_small = np.ascontiguousarray(target_small.astype(np.float32))
    source_small = itk.GetImageFromArray(source_small) 
    target_small = itk.GetImageFromArray(target_small)
    source_small.SetSpacing(spacing_small)
    target_small.SetSpacing(spacing_small)
    source_small.SetOrigin(origin_small)
    target_small.SetOrigin(origin_small)
    try:
        coreg_small, result_transform_parameters = itk.elastix_registration_method(
            target_small, source_small,
            parameter_object=params_obj, 
            log_to_console=log)
    except:
        warnings.warn('Elastix coregistration failed. Returning zero '
                      'deformation field. To find out the error, set log=True.')
        deformation_field = np.zeros(source_large.shape + (len(source_large.shape), ))
        return source_large.copy(), deformation_field
    
    # Get coregistered image at original size
    large_shape_x, large_shape_y = source_large.shape
    result_transform_parameters.SetParameter(0, "Size", [str(large_shape_y), str(large_shape_x)])
    result_transform_parameters.SetParameter(0, "Spacing", [str(spacing_large_y), str(spacing_large_x)])
    source_large = np.ascontiguousarray(source_large.astype(np.float32))
    source_large = itk.GetImageFromArray(source_large)
    source_large.SetSpacing(spacing_large)
    source_large.SetOrigin(origin_large)
    coreg_large = itk.transformix_filter(
        source_large,
        result_transform_parameters,
        log_to_console=log)
    coreg_large = itk.GetArrayFromImage(coreg_large)
    
    # Get deformation field at original size
    target_large = np.ascontiguousarray(target_large.astype(np.float32))
    target_large = itk.GetImageFromArray(target_large)
    target_large.SetSpacing(spacing_large)
    target_large.SetOrigin(origin_large)
    deformation_field = itk.transformix_deformation_field(
        target_large, 
        result_transform_parameters, 
        log_to_console=log)
    deformation_field = itk.GetArrayFromImage(deformation_field)
    deformation_field = np.reshape(deformation_field, target_large.shape + (len(target_large.shape), ))

    return coreg_large, deformation_field


def _coreg_3d(source_large, target_large, spacing, downsample, log, mask, 
              params_obj):

    if np.isscalar(spacing):
        spacing = [spacing, spacing, spacing]

    # Downsample source and target
    # The origin of an image is the center of the voxel in the lower left corner
    # The origin of the large image is (0,0,0).
    # The original of the small image is therefore: 
    #   spacing_large/2 + (spacing_small/2 - spacing_large)
    #   = (spacing_small - spacing_large)/2
    target_small = block_reduce(target_large, block_size=downsample, func=np.mean)
    source_small = block_reduce(source_large, block_size=downsample, func=np.mean)

    spacing_large = [spacing[2], spacing[1], spacing[0]] # correct numpy(x,y,z) ordering for itk(z,y,x)
    spacing_small = [spacing_large[0]*downsample, spacing_large[1]*downsample, spacing_large[2]*downsample] # downsample large spacing

    spacing_large_z, spacing_large_y, spacing_large_x = spacing_large # seperate for straightforward readable implementation with ITK ordering
    spacing_small_z, spacing_small_y, spacing_small_x = spacing_small

    large_shape_x, large_shape_y, large_shape_z = source_large.shape

    origin_large = [0,0,0]
    origin_small = [(spacing_small_z - spacing_large_z)/2, (spacing_small_y - spacing_large_y) / 2, (spacing_small_x - spacing_large_x) / 2]

    # Coregister downsampled source to target
    source_small = np.ascontiguousarray(source_small.astype(np.float32))
    target_small = np.ascontiguousarray(target_small.astype(np.float32))
    source_small = itk.GetImageFromArray(source_small) 
    target_small = itk.GetImageFromArray(target_small) 
    source_small.SetSpacing(spacing_small)
    target_small.SetSpacing(spacing_small)
    source_small.SetOrigin(origin_small)
    target_small.SetOrigin(origin_small)
    try:
        coreg_small, result_transform_parameters = itk.elastix_registration_method(
            target_small, source_small,
            parameter_object=params_obj, 
            log_to_console=log) # perform registration of downsampled image
    except:
        warnings.warn('Elastix coregistration failed. Returning zero '
                      'deformation field. To find out the error, set log=True.')
        deformation_field = np.zeros(source_large.shape + (len(source_large.shape), ))
        return source_large.copy(), deformation_field
    
    # Get coregistered image at original size
    result_transform_parameters.SetParameter(0, "Size", [str(large_shape_z), str(large_shape_y), str(large_shape_x)])
    result_transform_parameters.SetParameter(0, "Spacing", [str(spacing_large_z), str(spacing_large_y), str(spacing_large_x)])
    source_large = np.ascontiguousarray(source_large.astype(np.float32))
    source_large = itk.GetImageFromArray(source_large)
    source_large.SetSpacing(spacing_large)
    source_large.SetOrigin(origin_large)
    coreg_large = itk.transformix_filter(
        source_large,
        result_transform_parameters,
        log_to_console=log)
    
    # Get deformation field
    target_large = np.ascontiguousarray(target_large.astype(np.float32))
    target_large = itk.GetImageFromArray(target_large)
    target_large.SetSpacing(spacing_large)
    target_large.SetOrigin(origin_large)
    deformation_field = itk.transformix_deformation_field(
        target_large, 
        result_transform_parameters, 
        log_to_console=log)
    deformation_field = itk.GetArrayFromImage(deformation_field)
    deformation_field = np.reshape(deformation_field, target_large.shape + (len(target_large.shape), ))

    return coreg_large, deformation_field


def _cleanup(
        WriteResultImage = 'true',
        WriteDeformationField = "true", 
        **params):
    
    if WriteDeformationField == 'false':

        try:
            os.remove('deformationField.nii')
        except OSError:
            pass
        try:
            os.remove('deformationField.mhd')
        except OSError:
            pass
        try:
            os.remove('deformationField.raw')
        except OSError:
            pass

        path = os.path.dirname(__main__.__file__)

        try:
            os.remove(os.path.join(path, 'deformationField.nii'))
        except OSError:
            pass
        try:
            os.remove(os.path.join(path, 'deformationField.mhd'))
        except OSError:
            pass
        try:
            os.remove(os.path.join(path, 'deformationField.raw'))
        except OSError:
            pass
