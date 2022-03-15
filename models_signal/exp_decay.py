#!/ur/bin/python
# -*- coding: utf-8 -*-
"""
@author: Steven Sourbron 
Pixel-by-pixel exponential decay fit 
2022
"""

import numpy as np
from scipy.optimize import curve_fit


def func(t, S, R):
    """Mono-exponential decay function.  

    Args
    ----
    t (numpy.ndarray): x-values.  

    Returns
    -------
    numpy.ndarray: S*exp(-t*R)

    """
    return S*np.exp(-t*R)


def main(images, t, 
    lower_bounds = [0,0], 
    upper_bounds = [np.inf, np.inf], 
    initial_value = [1,1], 
    method = 'trf', 
    maxfev = 500, 
    ):
    """ main function that performs the T2*-map signal model-fit for input 2D image at multiple time-points (TEs).

    Args
    ----
    images (numpy.ndarray): input image at all time-series (i.e. at each TE time) with shape [x-dim*y-dim, total time-series].  
    t (list): list containing time points of exponential.  

    Returns
    -------
    fit (numpy.ndarray): signal model fit per pixel for whole image with shape [x-dim*y-dim, total time-series].  
    par (numpy.ndarray): output signal model fit parameters 'S' and 'R' stored in a single nd-array with shape [2, x-dim*y-dim].      
    """

    t = np.array(t)
    shape = np.shape(images)
    par = np.empty((2, shape[0])) # pixels should be first for consistency
    fit = np.empty(shape)

    for x in range(shape[0]):

        signal = images[x,:]
        par[:,x], _ = curve_fit(func, 
            xdata = t, 
            ydata = signal, 
            p0 = [np.max(signal)*initial_value[0], initial_value[1]], 
            bounds = (lower_bounds, upper_bounds), 
            method = method, 
            maxfev = maxfev, 
        )
        fit[x,:] = func(t, par[0,x], par[1,x])
  
    return fit, par

