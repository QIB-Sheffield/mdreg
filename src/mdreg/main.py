
import time
import numpy as np

from mdreg import elastix, skimage, utils, plot, models


def fit(moving: np.ndarray,
        fit_image = None,
        fit_coreg = None,
        fit_pixel = None,
        precision = 1.0,
        maxit = 3,
        verbose = 0,
        plot_params = None,
        force_2d = False,
    ):

    """
    Remove motion from a series of images or volumes.

    Parameters
    ----------
    moving : numpy.ndarray
        The series of images to be corrected, with dimensions (x,y,t) or 
        (x,y,z,t). 
    fit_image : dict or list, optional
        A dictionary defining the signal model. For a 
        slice-by-slice computation (4D array with force_2d=True), this can be 
        a list of dictionaries, one for each slice. If fit_image is not 
        provided, a constant model is used.  
    fit_coreg : dict, optional
        The parameters for coregistering the images. The default is None, which 
        uses bspline coregistration in elastix with default parameters. 
    fit_pixel : dict, optional
        A dictionary defining a single-pixel signal model.
        For a slice-by-slice computation (4D array with force_2d=True), this 
        can be a list of dictionaries, one for each slice. 
        The default is None.
    precision : float, optional
        The precision of the coregistration. The default is 1.0.
    maxit : int, optional
        The maximum number of iterations. The default is 3.
    verbose : int, optional
        The level of feedback to provide to the user. 0: no feedback; 1: text 
        output only; 2: text output and progress bars; 3: text output, progress 
        bars and image exports. The default is 0.
    plot_params : dict, optional
        The parameters for plotting the images when verbose = 3. Any keyword 
        arguments accepted by `mdreg.plot_series` can be included. 
        This keyword is ignored when verbose < 3. 
    force_2d : bool, optional
        By default, a 3-dimensional moving array will be coregistered with a 
        3-dimensional deformation field. To perform slice-by-slice 
        2-dimensional registration instead, set *force_2d* to True. This 
        keyword is ignored when the arrays are 2-dimensional. The 
        default is False.
    
    Returns
    -------
    coreg : numpy.ndarray
        The coregistered images with the same dimensions as *moving*.
    defo : numpy.ndarray
        The deformation field with the same dimensions as *moving*, and one 
        additional dimension for the components of the vector field. If 
        *moving* 
        has dimensions (x,y,t) and (x,y,z,t), then the deformation field will 
        have dimensions (x,y,2,t) and (x,y,z,3,t), respectively.
    fit : numpy.ndarray
        The fitted signal model with the same dimensions as *moving*.
    pars : dict
        The parameters of the fitted signal model with dimensions (x,y,n) or 
        (x,y,z,n), where n is the number of free parameters of the signal 
        model.
 
    """
    if fit_image is None:
        fit_image = {'func': models.constant}

    # 2D slice-by-slice coregistration
    
    if moving.ndim==4:
        if force_2d:
            coreg = np.zeros(moving.shape)
            defo = np.zeros(moving.shape[:3] + (2, moving.shape[3]))
            fit_array = np.zeros(moving.shape)
            for k in range(moving.shape[2]):
                if verbose > 0:
                    print('-----------------')
                    print('Fitting slice ' + str(k).zfill(3) )
                    print('-----------------')
                if isinstance(fit_image, dict):
                    fit_image_k = fit_image
                else:
                    fit_image_k = fit_image[k]
                if fit_pixel is None:
                    fit_pixel_k = None
                elif isinstance(fit_pixel, dict):
                    fit_pixel_k = fit_pixel
                else:
                    fit_pixel_k = fit_pixel[k]

                coreg[:,:,k,:], defo[:,:,k,:,:], fit_array[:,:,k,:], pars_k = fit(
                    moving[:,:,k,:],
                    fit_pixel = fit_pixel_k,
                    fit_image = fit_image_k,
                    fit_coreg = fit_coreg,
                    precision = precision,
                    maxit = maxit,
                    verbose = verbose,
                    plot_params = plot_params,
                )
                if k == 0:
                    pars = np.zeros(moving.shape[:3] + (pars_k.shape[2],))
                pars[:,:,k,:] = pars_k
            return coreg, defo, fit_array, pars

    # 2D or 3D coregistration   
    if not isinstance(fit_image, dict):
        raise ValueError(
            'For 3D coregistration, the fit_image argument must be a '
            'dictionary. ')

    if fit_coreg is None:
        fit_coreg = {'package': 'elastix'}
    
    if 'package' not in fit_coreg:
        fit_coreg['package'] = 'elastix'

    if plot_params is None:
        plot_params = {}

    coreg, defo = utils._init_output(moving)
    converged = False
    it = 1
    start = time.time()

    while not converged: 

        startit = time.time()

        # Fit signal model
        if verbose > 0:
            print('Fitting signal model (iteration ' + str(it) + ')')
        if fit_pixel is not None:
            fit_array, pars = utils.fit_pixels(coreg, **fit_pixel)
        else:
            fit_func = fit_image['func']
            kwargs = {i:fit_image[i] for i in fit_image if i!='func'}
            fit_array, pars = fit_func(coreg, **kwargs)

        # Fit deformation
        if verbose > 0:
            print('Fitting deformation field (iteration ' + str(it) + ')')
        
        fit_coreg['progress_bar'] = verbose>1
        coreg, defo_new = _coreg_series(moving, fit_array, **fit_coreg)

        # Check convergence
        corr = np.amax(np.linalg.norm(defo-defo_new, axis=-2))
        converged = corr <= precision 
        defo = defo_new
        
        if verbose > 0:
            print('Deformation correction after iteration ' + str(it) + ': ' + str(corr) + ' pixels')
            print('Calculation time for iteration ' + str(it) + ': ' + str((time.time()-startit)/60) + ' min')  
        if verbose==3:
            anim = plot.plot_series(moving, fit_array, coreg, filename='mdreg['+str(it)+']', **plot_params)

        if it == maxit: 
            break

        it += 1 

    if verbose > 0:
        print('Total calculation time: ' + str((time.time()-start)/60) + ' min')

    return coreg, defo, fit_array, pars



def _coreg_series(moving, fit, package='elastix', **fit_coreg):

    if package == 'elastix':
        fit_coreg = _set_mdreg_elastix_defaults(fit_coreg)
        return elastix.coreg_series(moving, fit, **fit_coreg)
    
    elif package == 'skimage':
        return skimage.coreg_series(moving, fit, **fit_coreg)
    
    else:
        raise NotImplementedError(
            'This coregistration package is not implemented')
    


def _set_mdreg_elastix_defaults(params):

    if "WriteResultImage" not in params:
        params["WriteResultImage"] = "false"
    if "WriteDeformationField" not in params:
        params["WriteDeformationField"] = "false"
    if "ResultImagePixelType" not in params:
        params["ResultImagePixelType"] = "float"

    # # Removing this for v0.4.2 as results appear to be worse
    # if 'Metric' not in params:
    #     params["Metric"] = "AdvancedMeanSquares"

    # # Settings pre v0.4.0 - unclear why - removed for now
    # if "FinalGridSpacingInPhysicalUnits" not in params:
    #     params["FinalGridSpacingInPhysicalUnits"] = "50.0"
    # if "AutomaticParameterEstimation" not in params:
    #     params["AutomaticParameterEstimation"] = "true"
    # if "ASGDParameterEstimationMethod" not in params:
    #     params["ASGDParameterEstimationMethod"] = "Original"
    # if "MaximumStepLength" not in params:
    #     params["MaximumStepLength"] = "1.0"
    # if "CheckNumberOfSamples" not in params:
    #     params["CheckNumberOfSamples"] = "true"
    # if "RandomCoordinate" not in params:
    #     params["ImageSampler"] = "RandomCoordinate"

    return params
    

