#!/ur/bin/python
# -*- coding: utf-8 -*-
"""
@author: Kanishka Sharma
T2-mapping signal model-fit
2021
"""

import numpy as np
from scipy.optimize import curve_fit

def exp_func(T2_prep_times,S0,T2):
    """ mono-exponential decay function performing T2-map fitting.

    Args
    ----
    T2_prep_times (numpy.ndarray): T2-preparation times in the T2-mapping sequence as input for the signal model fit  

    Returns
    -------
    S0, T2 (numpy.ndarray): output signal model fit parameters.  
    """

    return S0*np.exp(-T2_prep_times/T2)



def T2_fitting(images_to_be_fitted, T2_prep_times):
    """ curve fit function which returns the fit and fitted params S0 and T2.

    Args
    ----
    images_to_be_fitted (numpy.ndarray): input image at all time-series (i.e. at each T2-prep time) with shape [x-dim*y-dim, total time-series]  
    T2_prep_times (list): list containing T2-preparation times as input (independent variable) for the signal model fit  

    Returns
    -------
    fit (numpy.ndarray): signal model fit at all time-series with shape [x-dim*y-dim, total time-series]  
    S0 (numpy.ndarray): fitted parameter 'S0' with shape [x-dim*y-dim]  
    T2 (numpy.ndarray): fitted parameter 'T2' (ms) with shape [x-dim*y-dim].  
    """
    
    lb = [0,0]
    ub = [np.inf,np.inf]
    initial_guess = [np.max(images_to_be_fitted),80] 
    shape = np.shape(images_to_be_fitted)

    S0 = np.empty(shape[0]) 
    T2 = np.empty(shape[0])
    fit = np.empty(shape)

    for x in range(shape[0]):#pixels (x-dim*y-dim)
       popt, pcov = curve_fit(exp_func, xdata = T2_prep_times, ydata = images_to_be_fitted[x,:], p0=initial_guess, bounds=(lb,ub), method='trf')
       S0[x] =  popt[0] 
       T2[x] =  popt[1] 
       
    for x in range(shape[0]): # pixels (x-dim*y-dim)
       for i in range(shape[1]): # time-series (i)
           fit[x,i] = exp_func(T2_prep_times[i], S0[x], T2[x])
    
    return fit, S0, T2



def main(images_to_be_fitted, signal_model_parameters): 
    """ main function that performs the T2-map signal model-fit for input 2D image at multiple time-points. 

    Args
    ----
    images_to_be_fitted (numpy.ndarray): input image at all time-series (i.e. at each T2-preparation time) with shape [x-dim*y-dim, total time-series]  
    signal_model_parameters (list): list containing T2-prep times as list elements  

    Returns
    -------
    fit (numpy.ndarray): signal model fit at all time-series (i.e. at each T2-preparation time) with shape [x-dim*y-dim, total time-series]  
    fitted_parameters (numpy.ndarray): output signal model fit parameters 'S0' and 'T2' stored in a single nd-array with shape [2, x-dim*y-dim].   
    """

    T2_prep_times = signal_model_parameters 
 
    results = T2_fitting(images_to_be_fitted, T2_prep_times)
 
    fit = results[0]
    S0 = results[1]
    T2 = results[2]

    fitted_parameters_tuple = (S0, T2) # (2, 147456)
    fitted_parameters = np.vstack(fitted_parameters_tuple)

    return fit, fitted_parameters




