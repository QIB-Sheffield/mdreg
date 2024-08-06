
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
    """_summary_

    Args:
        moving (_type_): _description_
        fit_pixel (_type_, optional): _description_. Defaults to None.
        fit_image (dict, optional): _description_. Defaults to { 'func': models.constant, }.
        fit_coreg (dict, optional): _description_. Defaults to {}.
        precision (float, optional): _description_. Defaults to 1.0.
        maxit (int, optional): _description_. Defaults to 3.
        verbose (int, optional): the level of feedback to provide to the user. 0: no feedback; 1: text output only; 2: text output and progress bars; 3: text output, progress bars and image exports. Defaults to 0.
        plot_params (dict, optional): _description_. Defaults to {}.

    Returns:
        _type_: _description_
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
            fit, pars = utils.fit_pixels(coreg, **fit_pixel)
        else:
            fit_func = fit_image['func']
            kwargs = {i:fit_image[i] for i in fit_image if i!='func'}
            fit, pars = fit_func(coreg, **kwargs)

        # Fit deformation
        if verbose > 0:
            print('Fitting deformation field (iteration ' + str(it) + ')')
        if 'package' not in fit_coreg:
            fit_coreg['package'] = 'elastix'
        coreg, defo_new = _coreg_series(moving, fit, **fit_coreg)

        # Check convergence
        corr = np.amax(np.linalg.norm(defo-defo_new, axis=-2))
        converged = corr <= precision 
        defo = defo_new
        
        if verbose > 0:
            print('Deformation correction after iteration ' + str(it) + ': ' + str(corr) + ' pixels')
            print('Calculation time for iteration ' + str(it) + ': ' + str((time.time()-startit)/60) + ' min')  
        if verbose==3:
            plot.plot_series(moving, fit, coreg, filename='mdreg['+str(it)+']', **plot_params)

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
