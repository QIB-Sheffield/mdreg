""" 
@author: Kanishka Sharma  
T1-MOLLI model fit  
2021  
"""

import numpy as np
import os
import sys  
from scipy.optimize import curve_fit
import multiprocessing
np.set_printoptions(threshold=sys.maxsize)


def pars():
    return ['T1','T1_apparent','b','a']

def bounds():
    lower = [0, 0, 1.0, 1.0] 
    upper = [3000, 3000, np.inf, np.inf]
    return lower, upper

def exp_func(x, a, b, T1):
    """ exponential function for T1-fitting.

    Args
    ----
    x (numpy.ndarray): Inversion times (TI) in the T1-mapping sequence as input for the signal model fit.    
    
    Returns
    -------
    a, b, T1 (numpy.ndarray): signal model fitted parameters.  
    """

    return np.abs(a - b * np.exp(-x/T1)) 



def T1_fitting(images_to_be_fitted, inversion_times):
    """ Calls T1_fitting_pixel which contains the curve_fit function and returns the fit, the fitted parameters A,B, and T1.  

    Args
    ----
    images_to_be_fitted (numpy.ndarray): input image at all time-series (i.e. at each TI time) with shape [x-dim*y-dim, total time-series].    
    inversion_times (list): list containing T1 inversion times as input (independent variable) for the signal model fit.    

    Returns
    -------
    fit (numpy.ndarray): signal model fit at all time-series with shape [x-dim*y-dim, total time-series].
    T1_estimated, T1_apparent, b, a (numpy.ndarray): fitted parameters each with shape [x-dim*y-dim].    
    """

    lb = [0, 0, 0]
    ub = [np.inf, np.inf, 2000]
    p0 = [687.0, 1329.0, 1500.0]

    shape = np.shape(images_to_be_fitted)

    a = np.empty(shape[0]) 
    b = np.empty(shape[0])
    T1_apparent = np.empty(shape[0])
    fit = np.empty(shape)

    shape = np.shape(images_to_be_fitted)
    images_to_be_fitted = np.abs(images_to_be_fitted)

    try: 
        num_workers = int(len(os.sched_getaffinity(0)))
    except: 
        num_workers = int(os.cpu_count())

    # Run T1_fitting_pixel (which contains "curve_fit") in parallel processing
    pool = multiprocessing.Pool(processes=num_workers)
    arguments = [(x, images_to_be_fitted, p0, lb, ub, inversion_times) for x in range(shape[0])] #pixels (x-dim*y-dim)
    results = pool.map(T1_fitting_pixel, arguments)
    for i, result in enumerate(results):
        a[i] = result[0]
        b[i] = result[1]
        T1_apparent[i] = result[2]
        fit[i] = result[3]
    
    pool.close()
    pool.join()   
    
    T1_estimated = T1_apparent * (np.divide(b, a, out=np.zeros_like(b), where=a!=0) - 1)
    T1_estimated[T1_estimated>3000] = 3000
       
    return fit, T1_estimated, T1_apparent, b, a


def T1_fitting_pixel(parallel_arguments):
    """ Runs the curve fit function for 1 pixel.

    Args
    ----
    parallel_arguments (tuple): tuple containing the input arguments for curve_fit, such as the input images, the inversion times and more for the signal model fit.
                                This tuple format is required for the success of the parallelisation process and consequent speed-up of the fitting.

    Returns
    -------
    fitx (numpy.ndarray): signal model fit at all time-series with shape [total time-series].
    T1_apparent_x, bx, ax (numpy.ndarray): fitted parameters each with shape [1].
    """
    pixel_index, images_to_be_fitted, p0, lb, ub, inversion_times = parallel_arguments
    popt, pcov = curve_fit(exp_func, xdata = inversion_times, ydata = images_to_be_fitted[pixel_index,:], p0 = p0, bounds = (lb, ub), method = 'trf', maxfev=1000000) 
    ax =  popt[0] 
    bx =  popt[1] 
    T1_apparent_x = popt[2]
    fitx = []
    for i in range(len(inversion_times)):
        fitx.append(exp_func(inversion_times[i], ax, bx, T1_apparent_x))
    return ax, bx, T1_apparent_x, fitx


def main(images_to_be_fitted, signal_model_parameters): 
    """ main function that performs the T1-map signal model-fit for input 2D image at multiple time-points. 

        Args
        ----
        images_to_be_fitted (numpy.ndarray): input image at all time-series (i.e. at each TI time) with shape [x-dim*y-dim, total time-series].    
        signal_model_parameters (list): list containing T1 inversion times as list elements.    

        Returns
        -------
        fit (numpy.ndarray): signal model fit at all time-series (i.e. at each T1 inversion time) with shape [x-dim*y-dim, total time-series].   
        fitted_parameters (numpy.ndarray): output signal model fit parameters 'T1_estimated', 'T1_apparent', 'B', 'A' stored in a single nd-array with shape [4, x-dim*y-dim].     
    """
 
    inversion_times = signal_model_parameters 

    results = T1_fitting(images_to_be_fitted, inversion_times)
    
    fit = results[0]
    T1_estimated = results[1]
    T1_apparent = results[2]
    B = results[3]
    A = results[4]

    fitted_parameters_tuple = (T1_estimated, T1_apparent, B, A) 
    fitted_parameters = np.vstack(fitted_parameters_tuple)
    fitted_parameters = fitted_parameters.T

    return fit, fitted_parameters
