"""
@KanishkaS: modified for MDR-Library from previous implementation @Fotios Tagkalakis
"""

import numpy as np
import sys  
import pydicom
from scipy.optimize import curve_fit
np.set_printoptions(threshold=sys.maxsize)


### T1-MOLLI FUNCTION

def read_inversion_times_and_sort(fname,lstFilesDCM):
    """
    This function reads the inversion times for T1 sequences and sorts the files according to these inversion times.
    It takes as argument a list of dicoms as: fname,lstFilesDCM
    and returns a sorted list of inversion times and slices (as per the sorted inversion times)
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
    """
    Exponential function used to perform T1-fitting
    """
    return a - b * np.exp(-x/T1) #S = a-b*exp(-TI/T1)



def T1_fitting(images_to_be_fitted, inversion_times):

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
    


def fitting(images_to_be_fitted, signal_model_parameters):
    '''
    images_to_be_fitted: single_pixel at different inversion times
    signal model parameters for T1: [MODEL, inversion_times]
    '''
    fitted_parameters = []
   
    inversion_times = signal_model_parameters[1]

    fit, results = T1_fitting(images_to_be_fitted, inversion_times)
    
    T1_estimated = results[0]
    T1_apparent = results[1]
    B = results[2]
    A = results[3]
    
    fitted_parameters = [T1_estimated, T1_apparent, B, A]

    return fit, fitted_parameters