#!/ur/bin/python
# -*- coding: utf-8 -*-
"""
@author: Kanishka Sharma; iBEAt study; T2* mapping
"""

import numpy as np
import pydicom
from scipy.optimize import curve_fit

def read_echo_times(fname,lstFilesDCM):
    """
    This function reads the inversion times for T2* sequences and echo times
    It takes as argument as fname, lstFilesDCM
    and returns the sorted list of echo times and DICOMs from corresponding echo times
    """
    echo_times = []
    files = []

    for fname in lstFilesDCM:
        dataset = pydicom.dcmread(fname)   
        echo_times.append(dataset.EchoTime)
        files.append(pydicom.dcmread(fname)) 

    print(echo_times)

    sort_index = np.argsort(echo_times)
  
    echo_times.sort()
  
    slice_echo_time = []
    slice_sorted_echo_time = []

    for f in files: 
        slice_echo_time.append(f)

    # sorted slices using sorted echo times
    for i in range(0, len(slice_echo_time)):
         slice_sorted_echo_time.append(slice_echo_time[sort_index[i]])
   
    return echo_times, slice_sorted_echo_time



def exp_func(TE,S0,T2star):
    """
    mono-exponential function used to perform T2star-fitting
    """
  
    return S0*np.exp(-TE/T2star)



def T2star_fitting(images_to_be_fitted, echo_times):
    """
    curve fit function which returns the fit, and fitted params: S0 and T2*
    """
    
    lb = [0,10]
    ub = [np.inf,100]
    initial_guess = [np.max(images_to_be_fitted),50] 

    popt, pcov = curve_fit(exp_func, xdata = echo_times, ydata = images_to_be_fitted, p0=initial_guess, bounds=(lb,ub), method='trf')

    fit = []

    for te in echo_times:
        fit.append(exp_func(te, *popt))

    S0 = popt[0]
    T2star = popt[1]

    return fit, S0, T2star



def fitting(images_to_be_fitted, signal_model_parameters):
    '''
    main fitting function
    images_to_be_fitted: single_pixel at all echo times
    '''
    
    fitted_parameters = []
   
    echo_times = signal_model_parameters[1]

    results = T2star_fitting(images_to_be_fitted, echo_times)

    fit = results[0]
    S0 = results[1]
    T2star = results[2]
    
    fitted_parameters = [S0, T2star]

    return fit, fitted_parameters




