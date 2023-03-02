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

def bounds():
    lower = [0,0]
    upper = [np.inf, 100]
    return lower, upper

def pars():
    return ['S0', 'T2star']

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
    lb = [0,10]
    ub = [np.inf,100]
    initial_guess = [np.max(images_to_be_fitted),50] 
    shape = np.shape(images_to_be_fitted)
    S0 = np.empty(shape[0]) 
    T2_star = np.empty(shape[0])
    fit = np.empty(shape)

    # Run T2star_fitting_pixel (which contains "curve_fit") in parallel processing
    try: 
        num_workers = int(len(os.sched_getaffinity(0)))
    except: 
        num_workers = int(os.cpu_count())

    pool = multiprocessing.Pool(processes=num_workers)
    arguments = [(x, images_to_be_fitted, initial_guess, lb, ub, echo_times) for x in range(shape[0])] #pixels (x-dim*y-dim)
    results = pool.map(T2star_fitting_pixel, arguments)
    for i, result in enumerate(results):
        S0[i] = result[1]
        T2_star[i] = result[2]
        fit[i] = result[0]
  
    pool.close()
    pool.join()

    return fit, S0, T2_star


def T2star_fitting_pixel(parallel_arguments):
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
    pixel_index, images_to_be_fitted, initial_guess, lb, ub, echo_times = parallel_arguments
    #popt, pcov = curve_fit(exp_func, xdata = echo_times, ydata = images_to_be_fitted[pixel_index,:], p0=initial_guess, bounds=(lb,ub), method='trf')
    try:
        popt, pcov = curve_fit(exp_func, xdata = echo_times, ydata = images_to_be_fitted[pixel_index,:], p0=[np.max(images_to_be_fitted[pixel_index,:]), 60], bounds=(lb,ub), method='trf', maxfev=10000)
        S0x =  popt[0] 
        T2_star_x =  popt[1]
    except:
        S0x =  np.max(images_to_be_fitted[pixel_index,:])
        T2_star_x = 60

    fitx = []
    for i in range(len(echo_times)):#time-series (t)
        fitx.append(exp_func(echo_times[i], S0x, T2_star_x))
    return fitx, S0x, T2_star_x


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
    fitted_parameters = fitted_parameters.T

    return fit, fitted_parameters

