
"""
@author: Kanishka Sharma  
iBEAt study DWI monoexponential model fit for mdreg  
2021  
"""

import numpy as np
import os
import sys  
from scipy.optimize import curve_fit
import multiprocessing
np.set_printoptions(threshold=sys.maxsize)

def pars():
    return ['S0', 'ADC']

def bounds():
    lower = [0,0]
    upper = [np.inf, 1.0]
    return lower, upper


def func(b, S0, ADC):
    """ mono-exponential decay function used to perform DWI-fitting.  

    Args
    ----
    b (numpy.ndarray): b-values as input for the signal model fit.  

    Returns
    -------
    'S0' and 'ADC' (numpy.ndarray): fitted parameters S0 and ADC (apparent diffusion coefficent) (mm2/sec*10^-3) with shape [x-dim*y-dim].    
    """
 
    return S0*np.exp(-b*ADC)


def IVIM_fitting_pixel(args):
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
    S = args[0]
    b = args[1]
    nb = len(b)
    lb = [0,0]
    ub = [np.inf,1]
    p0 = [0.9*np.max(S),0.0025] 

    par = np.empty((3,2))
    try:
        par[0,:], _ = curve_fit(func, xdata = b, ydata = S[:nb], p0=p0, bounds=(lb,ub), method='trf', maxfev=500)
    except RuntimeError:
        par[0,:] = p0
    try:   
        par[1,:], _ = curve_fit(func, xdata = b, ydata = S[nb:2*nb], p0=p0, bounds=(lb,ub), method='trf', maxfev=500)
    except RuntimeError:
        par[1,:] = p0
    try:
        par[2,:], _ = curve_fit(func, xdata = b, ydata = S[2*nb:], p0=p0, bounds=(lb,ub), method='trf', maxfev=500)
    except RuntimeError:
        par[2,:] = p0

    fit = np.empty(3*nb)
    fit[:nb] = func(b, par[0,0], par[0,1])
    fit[nb:2*nb] = func(b, par[1,0], par[1,1])
    fit[2*nb:] = func(b, par[2,0], par[2,1])

    return fit, np.mean(par, axis=0)


def main(images, bvalues):
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
    bvalues = np.array(bvalues)
    shape = np.shape(images)
    fit = np.empty(shape)
    par = np.empty((shape[0], 2))

    pool = multiprocessing.Pool(processes=os.cpu_count()-1)
    arguments = [(images[x,:], bvalues) for x in range(shape[0])] 
    results = pool.map(IVIM_fitting_pixel, arguments)
    for x, result in enumerate(results):
        fit[x,:], par[x,:] = result
   
    return fit, par


