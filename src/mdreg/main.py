
import time
import numpy as np

from mdreg import elastix, skimage, utils, plot, models


def fit(moving,
        fit_pixel = None,
        fit_image = {
            'func': models.constant,
        },
        fit_coreg = {},
        precision = 1.0,
        maxit = 3,
        verbose = 0,
        plot_params = {},
    ):

    """
    Fit a signal model to a series of images and perform coregistration to correct for motion artefacts.

    Parameters
    ----------
    moving : numpy.array
        The series of images to be corrected.
        The array can be either 3D or 4D with the following shapes: 3D: 
        (X, Y, T). 4D: (X, Y, Z, T). Here, X, Y, Z are the spatial 
        dimensions and T is the dimension denoting change e.g. temporal 
        dimension or flip angle.
    fit_pixel : dict, optional
        The parameters for fitting the signal model to each pixel. The default 
        is None.
    fit_image : dict, optional
        The parameters for fitting the signal model to the whole image. The 
        default is {'func': models.constant, }.
    fit_coreg : dict, optional
        The parameters for coregistering the images. The default is {}.
    precision : float, optional
        The precision of the coregistration. The default is 1.0.
    maxit : int, optional
        The maximum number of iterations. The default is 3.
    verbose : int, optional
        The level of feedback to provide to the user. 0: no feedback; 1: text output only; 2: text output and progress bars; 3: text output, progress bars and image exports. The default is 0.
    plot_params : dict, optional
        The parameters for plotting the images. The default is {}.
    
    Returns
    -------
    coreg : numpy.array
        The coregistered images.
        The array matches the shape of the input moving array.
    defo : numpy.array
        The deformation field.
        The array matches the shape of the input moving array, with an 
        additional dimension showing deformation components. For 2D spatial
        images the deformation field has shape (X, Y, 2, T). For 3D spatial 
        images the deformation field has shape (X, Y, Z, 3, T).
    fit : numpy.array
        The fitted signal model.
        The array matches the shape of the input moving array.
    pars : dict
        The parameters of the fitted signal model.
        Array has the same shape as the spatial coordinates of the signal, 
        with a final extra axis length based on how many parameters the model 
        requires (N). For 2D spatial data: (X, Y, N). For 3D spatial data: 
        (X, Y, Z, N).

    """
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
            fit_pixel['progress_bar'] = verbose>1
            fit, pars = utils.fit_pixels(coreg, **fit_pixel)
        else:
            fit_func = fit_image['func']
            kwargs = {i:fit_image[i] for i in fit_image if i!='func'}
            kwargs['progress_bar'] = verbose>1
            fit, pars = fit_func(coreg, **kwargs)

        # Fit deformation
        if verbose > 0:
            print('Fitting deformation field (iteration ' + str(it) + ')')
        if 'package' not in fit_coreg:
            fit_coreg['package'] = 'elastix'
        
        fit_coreg['progress_bar'] = verbose>1
        coreg, defo_new = _coreg_series(moving, fit, **fit_coreg)

        # Check convergence
        corr = np.amax(np.linalg.norm(defo-defo_new, axis=-2))
        converged = corr <= precision 
        defo = defo_new
        
        if verbose > 0:
            print('Deformation correction after iteration ' + str(it) + ': ' + str(corr) + ' pixels')
            print('Calculation time for iteration ' + str(it) + ': ' + str((time.time()-startit)/60) + ' min')  
        if verbose==3:
            anim = plot.plot_series(moving, fit, coreg, filename='mdreg['+str(it)+']', **plot_params)

        if it == maxit: 
            break

        it += 1 

    if verbose > 0:
        print('Total calculation time: ' + str((time.time()-start)/60) + ' min')

    return coreg, defo, fit, pars



def _coreg_series(moving, fit, package='elastix', **coreg_params):

    if package == 'elastix':
        return elastix.coreg_series(moving, fit, **coreg_params)
    elif package == 'skimage':
        return skimage.coreg_series(moving, fit, **coreg_params)
    else:
        raise NotImplementedError('This coregistration package is not implemented')
