"""
@KanishkaS: modified for MDR-Library from previous implementation @Fotios Tagkalakis
@author: Kanishka Sharma
iBEAt study T1-MOLLI model fit
2021


    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!! UNDER CONSTRUCTION !!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!! DO NOT USE !!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



"""

import numpy as np
import sys  
import pydicom
from scipy.optimize import curve_fit
np.set_printoptions(threshold=sys.maxsize)


def read_inversion_times_and_sort(fname,lstFilesDCM):
    """ This function reads the inversion times for the T1 sequence and sorts the files according to these inversion times.
    
    Args
    ----
    fname (pathlib.PosixPath): dicom filenames to process
    lstFilesDCM (list): list of dicom files to process

    Returns
    -------
    inversion_times (list): sorted list of inversion times 
    slice_sorted_inv_time (list): sorted list of DICOMs from corresponding inversion times (TI).
    """

    inversion_times = []
    files = []
    for fname in lstFilesDCM:
        dataset = pydicom.dcmread(fname)   
        inversion_times.append(dataset.InversionTime)
        files.append(pydicom.dcmread(fname)) 
  
    sort_index = np.argsort(inversion_times)
  
    inversion_times.sort()

    slice_inv_time = []
    slice_sorted_inv_time = []

    for f in files: 
        slice_inv_time.append(f)

    # sorted array using sorted indices
    for i in range(0, len(slice_inv_time)):
         slice_sorted_inv_time.append(slice_inv_time[sort_index[i]])
   
    return inversion_times, slice_sorted_inv_time
    

def exp_func(x, a, b, T1):
    """ exponential function used to perform T1-fitting.

    Args
    ----
    x (list): list of inversion times (TI) 
    
    Returns
    -------
    a, b, S0, T1 (numpy.ndarray): signal model fitted parameters as np.ndarray.
    """

    return a - b * np.exp(-x/T1) #S = a-b*exp(-TI/T1)



def T1_fitting(images_to_be_fitted, inversion_times):
    """ curve fit function which returns the fit, and fitted params: S0 and apparent T1.



    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!! UNDER CONSTRUCTION !!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!! DO NOT USE !!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



    Args
    ----
    images_to_be_fitted (numpy.ndarray): pixel value for time-series (i.e. at each TI time) with shape [x,:]
    inversion_times (list): T1 inversion times

    Returns
    -------
    fit (list): signal model fit per pixel
    results (list): fitted parameters inluding 'T1_estimated' - estimated T1 (ms) and 'T1_apparent' - T1 (ms), and B, and A.
    """

    lb = [0, 0, 0]
    ub = [np.inf, np.inf, 2000]
    p0 = [np.max(images_to_be_fitted), np.max(images_to_be_fitted)-np.min(images_to_be_fitted), 50]

    #min_value = np.min(images_to_be_fitted)
  
    null_point = np.argmin(images_to_be_fitted)
 
    # convert to list and change sign of all elements before the null point
    images_to_be_fitted =images_to_be_fitted.tolist()

    for i in range(0, len(images_to_be_fitted)):
        if i < null_point:
           images_to_be_fitted[i] *= -1

    popt, pcov = curve_fit(exp_func, xdata = inversion_times, ydata = images_to_be_fitted, p0 = p0, bounds = (lb, ub), method = 'trf', maxfev=1000000) 

    
    fit = []
    for it in inversion_times:
        fit.append(exp_func(it, *popt))
      
    try:
        T1_estimated = popt[2] * ((popt[1] / popt[0]) - 1)
        if T1_estimated > 2000:
           T1_estimated = 0       

    except ZeroDivisionError:
        T1_estimated = 0
   
   # outliers outside expected range
    if popt[1] > 300:
       popt[1] = 0  

    A = popt[0]
    B = popt[1]
    T1_apparent = popt[2]

    results = []

    results = [T1_estimated, T1_apparent, B, A]

    return fit, results
    


def main(images_to_be_fitted, signal_model_parameters): 
    """ main function for model fitting at single pixel level. 



    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!! UNDER CONSTRUCTION !!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!! DO NOT USE !!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



        Args
        ----
        images_to_be_fitted (numpy.ndarray): pixel value for time-series (i.e. at each TI time) with shape [x,:]
        signal model parameters (list): T1 inversion times as a list 

        Returns
        -------
        fit (list): signal model fit per pixel
        fitted_parameters (list): list of estimated parameters from model fit including T1_estimated - estimated T1 (ms), 
        T1_apparent - apparent T1 (ms), and B, and A.
    """

    fitted_parameters = []
   
    inversion_times = signal_model_parameters 

    fit, results = T1_fitting(images_to_be_fitted, inversion_times)
    
    T1_estimated = results[0]
    T1_apparent = results[1]
    B = results[2]
    A = results[3]
    
    fitted_parameters = [T1_estimated, T1_apparent, B, A]

    return fit, fitted_parameters