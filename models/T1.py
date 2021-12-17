""" 
@author: Kanishka Sharma  
T1-MOLLI model fit  
2021  
"""

import numpy as np
import sys  
from scipy.optimize import curve_fit
np.set_printoptions(threshold=sys.maxsize)


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
    """ curve fit function which returns the fit, and fitted parameters A,B, and T1.  

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

    for x in range(shape[0]):#pixels(xdim*ydim)
        popt, pcov = curve_fit(exp_func, xdata = inversion_times, ydata = images_to_be_fitted[x,:], p0 = p0, bounds = (lb, ub), method = 'trf', maxfev=1000000) 
        
        a[x] =  popt[0] 
        b[x] =  popt[1] 
        T1_apparent[x] = popt[2]


    for x in range(shape[0]):
       for i in range(shape[1]):
           fit[x,i] = exp_func(inversion_times[i], a[x], b[x], T1_apparent[x])
       
    try:
        T1_estimated = T1_apparent * ((b / a) - 1)
       
    except ZeroDivisionError:
        T1_estimated = 0
    
    return fit, T1_estimated, T1_apparent, b, a
    

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

    return fit, fitted_parameters