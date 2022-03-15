
"""
@author: Kanishka Sharma  
iBEAt study DWI monoexponential model fit for the MDR Library  
2021  
"""

import numpy as np
import os
import sys  
from scipy.optimize import curve_fit
import multiprocessing
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


def IVIM_fitting_pixel(parallel_arguments):
    """ Runs the curve fit function for 1 pixel.

    Args
    ----
    parallel_arguments (tuple): tuple containing the input arguments for curve_fit, such as the input images, the bvalues and more for the signal model fit.
                                This tuple format is required for the success of the parallelisation process and consequent speed-up of the fitting.

    Returns
    -------
    fitx (numpy.ndarray): signal model fit at all time-series with shape [total time-series].  
    S0x, ADCx (numpy.ndarray): fitted parameters 'S0' and 'ADC' (mm2/sec*10^-3) each with shape [1].  
    """
    pixel_index, images_to_be_fitted, initial_guess, b_val, strt_idx, end_idx, lb, ub = parallel_arguments
    popt_x, pcov_x = curve_fit(exp_func, xdata = b_val[:10], ydata = images_to_be_fitted[pixel_index,:10], p0=initial_guess, bounds=(lb,ub), method='trf', maxfev=1000000)
    popt_y, pcov_y = curve_fit(exp_func, xdata = b_val[:10], ydata = images_to_be_fitted[pixel_index,strt_idx: end_idx], p0=initial_guess, bounds=(lb,ub), method='trf', maxfev=1000000)
    popt_z, pcov_z = curve_fit(exp_func, xdata = b_val[:10], ydata = images_to_be_fitted[pixel_index,-10:], p0=initial_guess, bounds=(lb,ub), method='trf', maxfev=1000000)
    S0_xx =  popt_x[0] 
    S0_yx =  popt_y[0]
    S0_zx =  popt_z[0]
    ADC_xx =  popt_x[1] 
    ADC_yx =  popt_y[1] 
    ADC_zx =  popt_z[1]
    fit_xx = []
    fit_yx = []
    fit_zx = []
    for i in range(10): # time-series with 10 b-vals per image series
        fit_xx.append(exp_func(b_val[i], S0_xx, ADC_xx))
        fit_yx.append(exp_func(b_val[i], S0_yx, ADC_yx))
        fit_zx.append(exp_func(b_val[i], S0_zx, ADC_zx))
    return fit_xx, fit_yx, fit_zx, S0_xx, S0_yx, S0_zx, ADC_xx, ADC_yx, ADC_zx


def IVIM_fitting(images_to_be_fitted, signal_model_parameters):
    """  Calls IVIM_fitting_pixel which contains the curve_fit function and returns the fit, and fitted params (S0 and ADC).  

    Args
    ----
    images_to_be_fitted (numpy.ndarray): input image at all time-series (i.e. at each b-value and for 3 acquired directions) with shape [x-dim*y-dim, total time-series].    
    inversion_times (list): list containing b-values as input (independent variable) for the signal model fit.    

    Returns
    -------
    fit (numpy.ndarray): signal model fit at all time-series with shape [x-dim*y-dim, total time-series].  
    S0, ADC (numpy.ndarray): fitted parameters 'S0' and 'ADC' (mm2/sec*10^-3) each with shape [x-dim*y-dim].  
    """

    b_val = signal_model_parameters # because we have 3 sets of repeated b vals
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

    # Run IVIM_fitting_pixel (which contains "curve_fit") in parallel processing
    pool = multiprocessing.Pool(processes=os.cpu_count()-1)
    arguments = [(x, images_to_be_fitted, initial_guess, b_val, strt_idx, end_idx, lb, ub) for x in range(shape[0])] #pixels (x-dim*y-dim)
    results = pool.map(IVIM_fitting_pixel, arguments)
    for i, result in enumerate(results):
        S0_x[i] = result[3]
        S0_y[i] = result[4]
        S0_z[i] = result[5]
        ADC_x[i] = result[6]
        ADC_y[i] = result[7]
        ADC_z[i] = result[8]
        fit_x[i,:] = result[0]
        fit_y[i,:] = result[1]
        fit_z[i,:] = result[2]
 
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
    
    results = IVIM_fitting(images_to_be_fitted, signal_model_parameters)

    fit = results[0]
    S0 = results[1]
    ADC = results[2]

    fitted_parameters_tuple = (S0, ADC) 
    fitted_parameters = np.vstack(fitted_parameters_tuple)

    return fit, fitted_parameters

