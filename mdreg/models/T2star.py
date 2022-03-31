#!/ur/bin/python
# -*- coding: utf-8 -*-
"""
@author: Kanishka Sharma  
T2*-mapping signal model-fit  
2021  
"""

import os
import numpy as np
from scipy.optimize import curve_fit
import multiprocessing

def pars():
    return ['S0', 'T2star']

def bounds():
    lower = [0,0]
    upper = [np.inf, 100]
    return lower, upper
    
def func(TE,S0,T2star):
    """ mono-exponential decay function to perform T2*-fitting.  

    Args
    ----
    TE (numpy.ndarray): Echo times (TE) in the T2*-mapping sequence as input (independent variable) for the signal model fit.  

    Returns
    -------
    S0, T2star(numpy.ndarray): output signal model fit parameters.  

    """
    return S0*np.exp(-TE/T2star)


def fit_pixel(args):
    """ Runs the curve fit function for 1 pixel.

    Args
    ----
    parallel_arguments (tuple): tuple containing the input arguments for curve_fit, such as the input images, the T2-preparation times and more for the signal model fit.
                                This tuple format is required for the success of the parallelisation process and consequent speed-up of the fitting.

    Returns
    -------
    fitx (numpy.ndarray): signal model fit at all time-series with shape [total time-series].    
    S0x (numpy.ndarray): fitted parameter 'S0' with shape [1].    
    T2x (numpy.ndarray): fitted parameter 'T2' (ms) with shape [1].  
    """
    S, TE = args
    p0 = [np.max(S),50]
    try:
        par, _ = curve_fit(func, 
            xdata = TE, 
            ydata = S, 
            p0 = p0, 
            bounds = ([0,10], [np.inf,100]),  
            method = 'trf', 
            maxfev = 500,
        )
    except RuntimeError:
        par = p0
    return func(TE, par[0], par[1]), par


def main(images, TE):
    """ Calls T2star_fitting_pixel which contains the curve_fit function and return the fit and fitted params S0 and T2*.

    Args
    ----
    images_to_be_fitted (numpy.ndarray):  input image at all time-series (i.e. at each TE time) with shape [x-dim*y-dim, total time-series].    
    echo_times (list): list containing TE times as input (independent variable) for the signal model fit.      

    Returns
    -------
    fit (numpy.ndarray): signal model fit at all time-series with shape [x-dim*y-dim, total time-series].      
    S0 (numpy.ndarray): fitted parameter 'S0' with shape [x-dim*y-dim].      
    T2_star (numpy.ndarray): fitted parameter 'T2_star' (ms) with shape [x-dim*y-dim].  

    """
    TE = np.array(TE)
    shape = np.shape(images)
    fit = np.empty(shape)
    par = np.empty((shape[0], 2))

    pool = multiprocessing.Pool(processes=os.cpu_count()-1)
    args = [(images[x,:], TE) for x in range(shape[0])] 
    results = pool.map(fit_pixel, args)
    for x, result in enumerate(results):
        fit[x,:], par[x,:] = result
        
    return fit, par