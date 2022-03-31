#!/ur/bin/python
# -*- coding: utf-8 -*-
"""
@author: Kanishka Sharma  
T1-mapping signal model-fit  
2021  
Messroghli DR, Radjenovic A, Kozerke S, Higgins DM, Sivananthan MU, Ridgway JP. 
Modified Look-Locker inversion recovery (MOLLI) for high-resolution T1 mapping of the heart. 
Magn Reson Med. 2004 Jul;52(1):141-6. doi: 10.1002/mrm.20110. PMID: 15236377. 
"""

import os
import numpy as np
from scipy.optimize import curve_fit
import multiprocessing

def pars():
    return ['A', 'B', 'T1app']

def bounds():
    lower = [0,0,0]
    upper = [np.inf, np.inf, 3000]
    return lower, upper

def func(x, a, b, T1):
    """ exponential function for T1-fitting.

    Args
    ----
    x (numpy.ndarray): Inversion times (TI) in the T1-mapping sequence as input for the signal model fit.    
    
    Returns
    -------
    a, b, T1 (numpy.ndarray): signal model fitted parameters.  
    """

    return np.abs(a - b * np.exp(-x/T1)) 


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

    p0 = [687.0, 1329.0, 1500.0]

    try:
        par, _ = curve_fit(func, 
            xdata = args[0], 
            ydata = args[1], 
            p0 = p0, 
            bounds = ([0, 0, 0], [np.inf, np.inf, 2000]),  
            method = 'trf', 
            maxfev = 500,
        )
    except RuntimeError: # optimum not found
        par = p0

    return func(args[0], par[0], par[1], par[2]), par


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
    par = np.empty((shape[0], 3))

    pool = multiprocessing.Pool(processes=os.cpu_count()-1)
    args = [(TE, images[x,:]) for x in range(shape[0])] 
    results = pool.map(fit_pixel, args)
    for x, result in enumerate(results):
        fit[x,:], par[x,:] = result
        
    return fit, par