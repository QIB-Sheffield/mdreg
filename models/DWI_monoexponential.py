
"""
@author: Kanishka Sharma  
iBEAt study DWI monoexponential model fit for the MDR Library  
2021  
"""

import numpy as np
import sys  
from scipy.optimize import curve_fit
np.set_printoptions(threshold=sys.maxsize)


def exp_func(b, S0, ADC):
    """ mono-exponential decay function used to perform DWI-fitting.  

    Args
    ----
    b (numpy.ndarray): b-values as input for the signal model fit.  

    Returns
    -------
    'S0' and 'ADC' (numpy.ndarray): fitted parameters S0 and ADC (apparent diffusion coefficent) (mm2/sec*10^-3) with shape [x-dim*y-dim].    
    """
 
    return S0*np.exp(-b*ADC)


def IVIM_fitting(images_to_be_fitted, signal_model_parameters):
    """ curve fit function which returns the fit, and fitted params (S0 and ADC).  

    Args
    ----
    images_to_be_fitted (numpy.ndarray): input image at all time-series (i.e. at each b-value and for 3 acquired directions) with shape [x-dim*y-dim, total time-series].    
    inversion_times (list): list containing b-values as input (independent variable) for the signal model fit.    

    Returns
    -------
    fit (numpy.ndarray): signal model fit at all time-series with shape [x-dim*y-dim, total time-series].  
    S0, ADC (numpy.ndarray): fitted parameters 'S0' and 'ADC' (mm2/sec*10^-3) each with shape [x-dim*y-dim].  
    """

    b_val = signal_model_parameters[0][0] # because we have 3 sets of repeated b vals
    lb = [0,0]
    ub = [np.inf,1]
    initial_guess = [0.9*np.max(images_to_be_fitted),0.0025] 
 
    K = 10 # 10 set of b-values per image series
    
    # computing strt, and end index for middle set of images 
    strt_idx = (len(images_to_be_fitted[0,:]) // 2) - (K // 2)
    end_idx = (len(images_to_be_fitted[0,:]) // 2) + (K // 2)
     
    shape = np.shape(images_to_be_fitted)
    S0_x, S0_y, S0_z = np.empty(shape[0]),  np.empty(shape[0]), np.empty(shape[0])
    ADC_x, ADC_y, ADC_z = np.empty(shape[0]), np.empty(shape[0]), np.empty(shape[0])
    fit_x, fit_y, fit_z = np.empty([shape[0],10]), np.empty([shape[0],10]), np.empty([shape[0],10])

    for x in range(shape[0]):#pixels (x*y)
       popt_x, pcov_x = curve_fit(exp_func, xdata = b_val[:10], ydata = images_to_be_fitted[x,:10], p0=initial_guess, bounds=(lb,ub), method='trf', maxfev=1000000)
       popt_y, pcov_y = curve_fit(exp_func, xdata = b_val[:10], ydata = images_to_be_fitted[x,strt_idx: end_idx], p0=initial_guess, bounds=(lb,ub), method='trf', maxfev=1000000)
       popt_z, pcov_z = curve_fit(exp_func, xdata = b_val[:10], ydata = images_to_be_fitted[x,-10:], p0=initial_guess, bounds=(lb,ub), method='trf', maxfev=1000000)
       S0_x[x] =  popt_x[0] 
       S0_y[x] =  popt_y[0]
       S0_z[x] =  popt_z[0]
       ADC_x[x] =  popt_x[1] 
       ADC_y[x] =  popt_y[1] 
       ADC_z[x] =  popt_z[1] 
       for i in range(10): # time-series with 10 b-vals per image series
           fit_x[x,i] = exp_func(b_val[i], S0_x[x], ADC_x[x])
           fit_y[x,i] = exp_func(b_val[i], S0_y[x], ADC_y[x])
           fit_z[x,i] = exp_func(b_val[i], S0_z[x], ADC_z[x])
 
    S0 = (S0_x + S0_y + S0_z)/3
    ADC = (ADC_x + ADC_y + ADC_z)/3
    fit = np.hstack((fit_x, fit_y, fit_z)) 
   
    return fit, S0, ADC


def main(images_to_be_fitted, signal_model_parameters):
    """ main function that performs the DWI signal model-fit for input 2D image at multiple time-points. 
 
    Args
    ----
    images_to_be_fitted (numpy.ndarray): input image at all time-series (i.e. at each b-value) with shape [x-dim*y-dim, total time-series].    
    signal_model_parameters (list): list containing 'b-values', 'b-vectors' and 'image_orientation_patient' as list elements.    

    Returns
    -------
    fit (numpy.ndarray): signal model fit at all time-series (i.e. at each b-value) with shape [x-dim*y-dim, total time-series].   
    fitted_parameters (numpy.ndarray): output signal model fit parameters 'S0' and 'ADC' stored in a single nd-array with shape [2, x-dim*y-dim].       
    """
    image_orientation_patient = signal_model_parameters[2] 
 
    for i in range(len(image_orientation_patient)-1):
        assert image_orientation_patient[i] == image_orientation_patient[i+1], "Error in image_orientation_patient for IVIM"
    
    results = IVIM_fitting(images_to_be_fitted, signal_model_parameters)

    fit = results[0]
    S0 = results[1]
    ADC = results[2]

    fitted_parameters_tuple = (S0, ADC) 
    fitted_parameters = np.vstack(fitted_parameters_tuple)

    return fit, fitted_parameters

