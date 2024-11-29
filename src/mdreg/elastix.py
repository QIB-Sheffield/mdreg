import os
import multiprocessing
import numpy as np
from tqdm import tqdm
import itk
from skimage.measure import block_reduce
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
            Coregistered series. 
    deformation : numpy.ndarray
            Deformation field.
    
    For more information on the main variables in terms of shape
    and description, see the :ref:`variable-types-table`.
    """
    if parallel:
        return _coreg_series_parallel(*args, **kwargs)
    else:
        return _coreg_series_sequential(*args, **kwargs)
    

def _coreg_series_sequential(source:np.ndarray, target:np.ndarray, 
        params = None,
        spacing = 1.0, 
        log = False, 
        mask = None, 
        downsample = 1,
        progress_bar = False,
    ):

    # This is a very slow step so needs to be done outside the loop
    p_obj = _make_params_obj(settings=params) 

    deformed, deformation = utils._init_output(source)
    for t in tqdm(range(source.shape[-1]), desc='Coregistering series', disable=not progress_bar): 

        if mask is not None:
            mask_t = mask[...,t]
        else: 
            mask_t = None

        deformed[...,t], deformation[...,t] = coreg(
                source[...,t], target[...,t], 
                params_obj=p_obj, 
                spacing=spacing, 
                log=log, 
                mask=mask_t,
                downsample=downsample,
            )
        
    return deformed, deformation


def _coreg_series_parallel(source:np.ndarray, target:np.ndarray, 
        params = None,
        spacing = 1.0, 
        log = False, 
        mask = None, 
        downsample = 1,
    ):

    # itk.force_load() # should speed up (but doesn't)
    # https://github.com/InsightSoftwareConsortium/ITKElastix/issues/204

    try: 
        num_workers = int(len(os.sched_getaffinity(0)))
    except: 
        num_workers = int(os.cpu_count())

    # Build list of arguments
    args = []
    for t in range(source.shape[-1]):
        if mask is not None:
            mask_t = mask[...,t]
        else: 
            mask_t = None
        args_t = (source[...,t], target[...,t], params, spacing, log, mask_t, downsample)
        args.append(args_t)

    # Process list of arguments in parallel
    pool = multiprocessing.Pool(processes=num_workers)
    results = pool.map(_coreg_parallel, args)
    # Good practice to close and join when the pool is no longer needed
    # https://stackoverflow.com/questions/38271547/when-should-we-call-multiprocessing-pool-join
    pool.close()
    pool.join()

    # Reformat list of results into arrays
    coreg, deformation = utils._init_output(source)
    for t in range(source.shape[-1]):
        coreg[...,t] = results[t][0]
        deformation[...,t] = results[t][1]

    return coreg, deformation


def _coreg_parallel(args):
    source, target, params, spacing, log, mask, downsample = args
    return coreg(source, target, params=params, spacing=spacing, log=log, mask=mask, downsample=downsample)


def coreg(source:np.ndarray, *args, **kwargs):

    """
    Coregister two arrays
    
    Parameters
    ----------
    source : numpy.ndarray
        The source image. 
        The array can be either 3D or 4D with the following shapes: 2D: (X, Y).
        3D: (X, Y, Z).
    *args : dict
        Coregistration arguments.
    **kwargs : dict
        Coregistration keyword arguments.
    
    Returns
    -------
    coreg : numpy.ndarray
        Coregistered image.
        The array can be either 3D or 4D with the following shapes: 2D: (X, Y). 
        3D: (X, Y, Z).
    deformation : numpy.ndarray
        Deformation field.
        The array can be either 3D or 4D with the following shapes: 3D: 
        (X, Y, 2). 4D: (X, Y, Z, 3).
    
    """

    if source.ndim == 2: 
        return _coreg_2d(source, *args, **kwargs)
    
    if source.ndim == 3:
        return _coreg_3d(source, *args, **kwargs)

    
def _coreg_2d(source_large, target_large, params=None, params_obj=None, spacing=1.0, log=False, mask=None, downsample=1):

    if np.isscalar(spacing):
        spacing = [spacing, spacing]

    if params_obj is None:
        params_obj = _make_params_obj(settings=params)
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
    source_small = itk.GetImageFromArray(np.array(source_small, np.float32)) 
    target_small = itk.GetImageFromArray(np.array(target_small, np.float32))
    source_small.SetSpacing(spacing_small)
    target_small.SetSpacing(spacing_small)
    source_small.SetOrigin(origin_small)
    target_small.SetOrigin(origin_small)
    coreg_small, result_transform_parameters = itk.elastix_registration_method(
        target_small, source_small,
        parameter_object=params_obj, 
        log_to_console=log)
    
    # Get coregistered image at original size
    large_shape_x, large_shape_y = source_large.shape
    result_transform_parameters.SetParameter(0, "Size", [str(large_shape_y), str(large_shape_x)])
    result_transform_parameters.SetParameter(0, "Spacing", [str(spacing_large_y), str(spacing_large_x)])
    source_large = itk.GetImageFromArray(np.array(source_large, np.float32))
    source_large.SetSpacing(spacing_large)
    source_large.SetOrigin(origin_large)
    coreg_large = itk.transformix_filter(
        source_large,
        result_transform_parameters,
        log_to_console=log)
    coreg_large = itk.GetArrayFromImage(coreg_large)
    
    # Get deformation field at original size
    target_large = itk.GetImageFromArray(np.array(target_large, np.float32))
    target_large.SetSpacing(spacing_large)
    target_large.SetOrigin(origin_large)
    deformation_field = itk.transformix_deformation_field(
        target_large, 
        result_transform_parameters, 
        log_to_console=log)
    deformation_field = itk.GetArrayFromImage(deformation_field).flatten()
    deformation_field = np.reshape(deformation_field, target_large.shape + (len(target_large.shape), ))

    return coreg_large, deformation_field


def _coreg_3d(source_large, target_large, params=None, params_obj=None, spacing=1.0, log=False, mask=None, downsample=1):

    if np.isscalar(spacing):
        spacing = [spacing, spacing, spacing]

    if params_obj is None:
        params_obj = _make_params_obj(settings=params)

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
    source_small = itk.GetImageFromArray(np.array(source_small, np.float32)) # convert to itk compatiable format
    target_small = itk.GetImageFromArray(np.array(target_small, np.float32)) # convert to itk compatiable format
    source_small.SetSpacing(spacing_small)
    target_small.SetSpacing(spacing_small)
    source_small.SetOrigin(origin_small)
    target_small.SetOrigin(origin_small)
    coreg_small, result_transform_parameters = itk.elastix_registration_method(
        target_small, source_small,
        parameter_object=params_obj, 
        log_to_console=log) # perform registration of downsampled image
    
    # Get coregistered image at original size
    result_transform_parameters.SetParameter(0, "Size", [str(large_shape_z), str(large_shape_y), str(large_shape_x)])
    result_transform_parameters.SetParameter(0, "Spacing", [str(spacing_large_z), str(spacing_large_y), str(spacing_large_x)])
    source_large = itk.GetImageFromArray(np.array(source_large, np.float32))
    source_large.SetSpacing(spacing_large)
    source_large.SetOrigin(origin_large)
    coreg_large = itk.transformix_filter(
        source_large,
        result_transform_parameters,
        log_to_console=log)
    
    # Get deformation field at original size
    target_large = itk.GetImageFromArray(np.array(target_large, np.float32))
    target_large.SetSpacing(spacing_large)
    target_large.SetOrigin(origin_large)
    deformation_field = itk.transformix_deformation_field(
        target_large, 
        result_transform_parameters, 
        log_to_console=log)
    
    deformation_field = itk.GetArrayFromImage(deformation_field)

    return coreg_large, deformation_field


def params(default='freeform', **override):

    """
    Generate parameters for elastix registration.

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

        
    Example:

        Adjust the default parameters associated with grid spacing for elastix 
        registration.

    .. plot::
        :include-source:
        :context: close-figs
    
        >>> import mdreg

        Adjust the default parameters associated with grid spacing for elastix 
        registration.

        >>> params = mdreg.params()
        >>> print(params['FinalGridSpacingInPhysicalUnits'])
        50.0

        Override the default parameters associated with grid spacing for 
        elastix registration.

        >>> params = mdreg.params(FinalGridSpacingInPhysicalUnits='5.0')
        >>> print(params['FinalGridSpacingInPhysicalUnits'])
        5.0
    """

    if default=='freeform':
        params = _freeform()
    else:
        raise ValueError('This default is not available')
    for key, val in override.items():
        params[key]=val
    return params

    
def _freeform():
    settings = {}
    # See here for default bspline settings and explanation of parameters
    # https://github.com/SuperElastix/ElastixModelZoo/tree/master/models%2Fdefault
    # *********************
    # * ImageTypes
    # *********************
    settings["FixedInternalImagePixelType"]="float"
    settings["MovingInternalImagePixelType"]="float"
    ## selection based on 3D or 2D image data: newest elastix version does not require input image dimension
    # settings["FixedImageDimension", d) 
    # settings["MovingImageDimension", d) 
    settings["UseDirectionCosines"]="true"
    # *********************
    # * Components
    # *********************
    settings["Registration"]="MultiResolutionRegistration"
    # Image intensities are sampled using an ImageSampler, Interpolator and ResampleInterpolator.
    # Image sampler is responsible for selecting points in the image to sample. 
    # The RandomCoordinate simply selects random positions.
    settings["ImageSampler"]="RandomCoordinate"
    # Interpolator is responsible for interpolating off-grid positions during optimization. 
    # The BSplineInterpolator with BSplineInterpolationOrder = 1 used here is very fast and uses very little memory
    settings["Interpolator"]="BSplineInterpolator"
    # ResampleInterpolator here chosen to be FinalBSplineInterpolator with FinalBSplineInterpolationOrder = 1
    # is used to resample the result image from the moving image once the final transformation has been found.
    # This is a one-time step so the additional computational complexity is worth the trade-off for higher image quality.
    settings["ResampleInterpolator"]="FinalBSplineInterpolator"
    settings["Resampler"]="DefaultResampler"
    # Order of B-Spline interpolation used during registration/optimisation.
    # It may improve accuracy if you set this to 3. Never use 0.
    # An order of 1 gives linear interpolation. This is in most 
    # applications a good choice.
    settings["BSplineInterpolationOrder"]="1"
    # Order of B-Spline interpolation used for applying the final
    # deformation.
    # 3 gives good accuracy; recommended in most cases.
    # 1 gives worse accuracy (linear interpolation)
    # 0 gives worst accuracy, but is appropriate for binary images
    # (masks, segmentations); equivalent to nearest neighbor interpolation.
    settings["FinalBSplineInterpolationOrder"]="3"
    # Pyramids found in Elastix:
    # 1)	Smoothing -> Smoothing: YES, Downsampling: NO
    # 2)	Recursive -> Smoothing: YES, Downsampling: YES
    #      If Recursive is chosen and only # of resolutions is given 
    #      then downsamlping by a factor of 2 (default)
    # 3)	Shrinking -> Smoothing: NO, Downsampling: YES
    settings["FixedImagePyramid"]="FixedRecursiveImagePyramid"
    settings["MovingImagePyramid"]="MovingRecursiveImagePyramid"
    settings["Optimizer"]="AdaptiveStochasticGradientDescent"
    # Whether transforms are combined by composition or by addition.
    # In generally, Compose is the best option in most cases.
    # It does not influence the results very much.
    settings["HowToCombineTransforms"]="Compose"
    settings["Transform"]="BSplineTransform"
    # Metric
    settings["Metric"]="AdvancedMeanSquares"
    # Number of grey level bins in each resolution level,
    # for the mutual information. 16 or 32 usually works fine.
    # You could also employ a hierarchical strategy:
    #(NumberOfHistogramBins 16 32 64)
    settings["NumberOfHistogramBins"]="32"
    # *********************
    # * Transformation
    # *********************
    # The control point spacing of the bspline transformation in 
    # the finest resolution level. Can be specified for each 
    # dimension differently. Unit: mm.
    # The lower this value, the more flexible the deformation.
    # Low values may improve the accuracy, but may also cause
    # unrealistic deformations.
    # By default the grid spacing is halved after every resolution,
    # such that the final grid spacing is obtained in the last 
    # resolution level.
    # The grid spacing here is specified in voxel units.
    #(FinalGridSpacingInPhysicalUnits 10.0 10.0)
    #(FinalGridSpacingInVoxels 8)
    #settings["FinalGridSpacingInPhysicalUnits", ["50.0", "50.0"])
    settings["FinalGridSpacingInPhysicalUnits"]="50.0"
    # *********************
    # * Optimizer settings
    # *********************
    # The number of resolutions. 1 Is only enough if the expected
    # deformations are small. 3 or 4 mostly works fine. For large
    # images and large deformations, 5 or 6 may even be useful.
    settings["NumberOfResolutions"]="4"
    settings["AutomaticParameterEstimation"]="true"
    settings["ASGDParameterEstimationMethod"]="Original"
    settings["MaximumNumberOfIterations"]="500"
    # The step size of the optimizer, in mm. By default the voxel size is used.
    # which usually works well. In case of unusual high-resolution images
    # (eg histology) it is necessary to increase this value a bit, to the size
    # of the "smallest visible structure" in the image:
    settings["MaximumStepLength"]="1.0" 
    # *********************
    # * Pyramid settings
    # *********************
    # The downsampling/blurring factors for the image pyramids.
    # By default, the images are downsampled by a factor of 2
    # compared to the next resolution.
    #settings["ImagePyramidSchedule"]="8 8  4 4  2 2  1 1"
    # *********************
    # * Sampler parameters
    # *********************
    # Number of spatial samples used to compute the mutual
    # information (and its derivative) in each iteration.
    # With an AdaptiveStochasticGradientDescent optimizer,
    # in combination with the two options below, around 2000
    # samples may already suffice.
    settings["NumberOfSpatialSamples"]="2048"
    # Refresh these spatial samples in every iteration, and select
    # them randomly. See the manual for information on other sampling
    # strategies.
    settings["NewSamplesEveryIteration"]="true"
    settings["CheckNumberOfSamples"]="true"
    # *********************
    # * Mask settings
    # *********************
    # If you use a mask, this option is important. 
    # If the mask serves as region of interest, set it to false.
    # If the mask indicates which pixels are valid, then set it to true.
    # If you do not use a mask, the option doesn't matter.
    settings["ErodeMask"]="false"
    settings["ErodeFixedMask"]="false"
    # *********************
    # * Output settings
    # *********************
    #Default pixel value for pixels that come from outside the picture:
    settings["DefaultPixelValue"]="0"
    # Choose whether to generate the deformed moving image.
    # You can save some time by setting this to false, if you are
    # not interested in the final deformed moving image, but only
    # want to analyze the deformation field for example.
    settings["WriteResultImage"]="true"
    # The pixel type and format of the resulting deformed moving image
    settings["ResultImagePixelType"]="float"
    settings["ResultImageFormat"]="mhd"
    
    return settings


def _make_params_obj(default='bspline', settings=None):

    """
    Make an elastix parameter object.

    Parameters
    ----------
    default : str
        The default parameter set to use.
    settings : dict
        Parameters to override.
    """
    param_obj = itk.ParameterObject.New() # long runtime ~20s
    parameter_map_bspline = param_obj.GetDefaultParameterMap(default) 
    param_obj.AddParameterMap(parameter_map_bspline) 
    if settings is None:
        return param_obj
    for p in settings:
        param_obj.SetParameter(p, settings[p])
    return param_obj










# Retired

def _OLD_coreg_2d(source, target, params_obj, spacing, log, mask):
    """
    Coregister two arrays and return coregistered + deformation field 
    """
    shape_source = np.shape(source)

    # Coregister source to target
    source = itk.GetImageFromArray(np.array(source, np.float32))
    target = itk.GetImageFromArray(np.array(target, np.float32))
    source.SetSpacing(spacing)
    target.SetSpacing(spacing)
    coregistered, result_transform_parameters = itk.elastix_registration_method(
        target, source, parameter_object=params_obj, log_to_console=log)
    coregistered = itk.GetArrayFromImage(coregistered)

    # Get deformation field
    deformation_field = itk.transformix_deformation_field(
        target, 
        result_transform_parameters, 
        log_to_console=log)
    deformation_field = itk.GetArrayFromImage(deformation_field).flatten()
    deformation_field = np.reshape(deformation_field, shape_source + (len(shape_source), ))

    return coregistered, deformation_field


