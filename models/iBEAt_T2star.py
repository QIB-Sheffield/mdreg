#!/ur/bin/python
# -*- coding: utf-8 -*-
"""
@author: Kanishka Sharma
iBEAt study T2* model fit
2021


    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!! UNDER CONSTRUCTION !!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!! DO NOT USE !!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


"""

import numpy as np
import pydicom
from scipy.optimize import curve_fit

def read_and_sort_echo_times(fname,lstFilesDCM):
    """ This function provides sorted list of DICOMs from a sorted list of T2* echo times (TE).


    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!! UNDER CONSTRUCTION !!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!! DO NOT USE !!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



    Args
    ----
    fname (pathlib.PosixPath): dicom filenames to process
    lstFilesDCM (list): list of dicom files to process

    Returns
    -------
    echo_times (list): sorted list of echo times 
    slice_sorted_echo_time (list): sorted list of DICOMs from sorted list of echo times (TE).

    """
    echo_times = []
    files = []

    for fname in lstFilesDCM:
        dataset = pydicom.dcmread(fname)   
        echo_times.append(dataset.EchoTime)
        files.append(pydicom.dcmread(fname)) 

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
    """ mono-exponential decay function used to perform T2star-fitting.

    Args
    ----
    TE (int): Echo times (TE) for per time-series point for the T2* mapping sequence

    Returns
    -------
    S0, T2star(numpy.ndarray): signal model fitted parameters as np.ndarray.

    """
  
    return S0*np.exp(-TE/T2star)



def T2star_fitting(images_to_be_fitted, echo_times):
    """ curve fit function which returns the fit and fitted params S0 and T2*.

    Args
    ----
    images_to_be_fitted (numpy.ndarray): pixel value for time-series (i.e. at each TE time) with shape [x,:]
    echo_times (list): list of TE times

    Returns
    -------
    fit (list): signal model fit per pixel
    S0 (numpy.float64): fitted parameter 'S0' per pixel 
    T2star (numpy.float64): fitted parameter 'T2*' (ms) per pixel.

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



def main(images_to_be_fitted, signal_model_parameters):
    """ main function for model fitting of T2* at single pixel level. 

    Args
    ----
    images_to_be_fitted (numpy.ndarray): pixel value for time-series (i.e. at each TE) with shape [x,:]
    signal_model_parameters (list): TE times as a list


    Returns
    -------
    fit (list): signal model fit per pixel
    fitted_parameters (list): list with signal model fitted parameters 'S0' and 'T2star'.  
    """
    
    echo_times = signal_model_parameters

    results = T2star_fitting(images_to_be_fitted, echo_times)

    fit = results[0]
    S0 = results[1]
    T2star = results[2]
    
    fitted_parameters = [S0, T2star]

    return fit, fitted_parameters




