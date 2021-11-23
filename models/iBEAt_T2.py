#!/ur/bin/python
# -*- coding: utf-8 -*-
"""
@author: Kanishka Sharma
iBEAt study T2-mapping model-fit
Siemens 3T PRISMA - Leeds
2021


    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!! UNDER CONSTRUCTION !!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!! DO NOT USE !!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


"""

import numpy as np
from scipy.optimize import curve_fit


def read_prep_times():
    """ This function manually reads T2-prep times (iBEAt study specific function) and returns it as a list."""
    ## hard coded as these are not available in the anonymised Siemens dicom tags
    T2_prep_times = [0,30,40,50,60,70,80,90,100,110,120]
    
    return T2_prep_times



def exp_func(T2_prep_times,S0,T2):
    """ mono-exponential decay function performing T2-fitting.

    Args
    ----
    T2_prep_times (int): T2-preparation time for per time-series point for the T2 mapping sequence

    Returns
    -------
    S0, T2 (numpy.ndarray): signal model fitted parameters in np.ndarray.
    
    """
  
    return S0*np.exp(-T2_prep_times/T2)



def T2_fitting(images_to_be_fitted, T2_prep_times):
    """ curve_fit function for T2-mapping.


    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!! UNDER CONSTRUCTION !!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!! DO NOT USE !!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



    Args
    ----
    images_to_be_fitted (numpy.ndarray): pixel value for time-series (i.e. at each T2-prep time) with shape [x,:]
    T2_prep_times (list): list of T2-preparation times

    Returns
    -------
    fit (list): signal model fit per pixel
    S0 (numpy.float64): fitted parameter 'S0' per pixel 
    T2 (numpy.float64): fitted parameter 'T2' (ms) per pixel.
    """
    
    lb = [0,0]
    ub = [np.inf,np.inf]
    initial_guess = [np.max(images_to_be_fitted),80] 

    popt, pcov = curve_fit(exp_func, xdata = T2_prep_times, ydata = images_to_be_fitted, p0=initial_guess, bounds=(lb,ub), method='trf')
    
    T2_prep_times
    fit = []

    for te in T2_prep_times:
        fit.append(exp_func(te, *popt))

    S0 = popt[0]
    T2 = popt[1]

    return fit, S0, T2



def main(images_to_be_fitted, signal_model_parameters):
    """ main function that performs the T2 model-fit at single pixel level. 


    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!! UNDER CONSTRUCTION !!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!! DO NOT USE !!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



    Args
    ----
    images_to_be_fitted (numpy.ndarray): pixel value for time-series (i.e. at each T2-prep time) with shape [x,:]
    signal_model_parameters (list): T2-prep times as a list


    Returns
    -------
    fit (list): signal model fit per pixel
    fitted_parameters (list): list with signal model fitted parameters 'S0' and 'T2'.  
    """

    T2_prep_times = signal_model_parameters
    results = T2_fitting(images_to_be_fitted, T2_prep_times)
 
    fit = results[0]
    S0 = results[1]
    T2 = results[2]
    
    fitted_parameters = [S0, T2]

    return fit, fitted_parameters




