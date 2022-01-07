#!/ur/bin/python
# -*- coding: utf-8 -*-
"""
@author: Kanishka Sharma  
T2*-mapping signal model-fit  
2021  
"""

import numpy as np
from scipy.optimize import curve_fit

def exp_func(TE,S0,T2star):
    """ mono-exponential decay function to perform T2*-fitting.  

    Args
    ----
    TE (numpy.ndarray): Echo times (TE) in the T2*-mapping sequence as input (independent variable) for the signal model fit.  

    Returns
    -------
    S0, T2star(numpy.ndarray): output signal model fit parameters.  

    """
  
    return S0*np.exp(-TE/T2star)



def T2star_fitting(images_to_be_fitted, echo_times):
    """ curve fit function which returns the fit and fitted params S0 and T2*.

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
    lb = [0,10]
    ub = [np.inf,100]
    initial_guess = [np.max(images_to_be_fitted),50] 
    shape = np.shape(images_to_be_fitted)
    S0 = np.empty(shape[0]) 
    T2_star = np.empty(shape[0])
    fit = np.empty(shape)

    for x in range(shape[0]):#pixels (x-dim*y-dim)
       popt, pcov = curve_fit(exp_func, xdata = echo_times, ydata = images_to_be_fitted[x,:], p0=initial_guess, bounds=(lb,ub), method='trf')
       S0[x] =  popt[0] 
       T2_star[x] =  popt[1]
       for i in range(shape[1]):#time-series (t)
          fit[x,i] = exp_func(echo_times[i], S0[x], T2_star[x])
  
    return fit, S0, T2_star



def main(images_to_be_fitted, signal_model_parameters):
    """ main function that performs the T2*-map signal model-fit for input 2D image at multiple time-points (TEs).

    Args
    ----
    images_to_be_fitted (numpy.ndarray): input image at all time-series (i.e. at each TE time) with shape [x-dim*y-dim, total time-series].  
    signal_model_parameters (list): list containing TE times as list elements.  

    Returns
    -------
    fit (numpy.ndarray): signal model fit per pixel for whole image with shape [x-dim*y-dim, total time-series].  
    fitted_parameters (numpy.ndarray): output signal model fit parameters 'S0' and 'T2star' stored in a single nd-array with shape [2, x-dim*y-dim].      
    """
    
    echo_times = signal_model_parameters
    results = T2star_fitting(images_to_be_fitted, echo_times)

    fit = results[0]
    S0 = results[1]
    T2star = results[2]

    fitted_parameters_tuple = (S0, T2star) 
    fitted_parameters = np.vstack(fitted_parameters_tuple) 

    return fit, fitted_parameters

