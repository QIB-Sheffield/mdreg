
"""
@author: Kanishka Sharma
iBEAt study DWI monoexponential model fit for the MDR Library
2021
"""

import numpy as np
import sys  
import pydicom
from itertools import repeat
from scipy.optimize import curve_fit
import functools
import operator
np.set_printoptions(threshold=sys.maxsize)

 
    
def read_dicom_tags_IVIM(fname,lstFilesDCM):
    """ This function reads the DICOM tags from the IVIM sequence and returns the corresponding DWI/IVIM tags.

    Args
    ----
    filenameDCM (pathlib.PosixPath): dicom filenames to process
    lstFilesDCM (list): list of dicom files to process

    Returns
    -------
    b-values (list): list of DWI/IVIM b-values (s/mm2) 
    b_Vec_original (list): original b-vectors as list
    image_orientation_patient (list):  patient orientation as list
    slice_sorted_b_values (list): list of slices sorted according to b-values
    """

    b_values = []
    b_values_sort = []
    b_Vec_original = []
    image_orientation_patient = []
    files = []
 
    g_dir_01 = [1,0,0]
    g_dir_02 = [0,1,0]
    g_dir_03 = [0,0,1]
    #TODO: to check gradient files
    for i in range(10):
        b_Vec_original.append(g_dir_01)

    for i in range(10,20):
        b_Vec_original.append(g_dir_02)

    for i in range(21,30):
        b_Vec_original.append(g_dir_03)

    #TODO: calculate from gradient file instead of manual list
    b_values = [0,10.000086, 19.99908294, 30.00085926, 50.00168544, 80.007135, 100.0008375, 199.9998135, 300.0027313, 600.0]
    b_values = list(repeat(b_values, 3))

    for fname in lstFilesDCM:
        dataset = pydicom.dcmread(fname)
      
        image_orientation_patient.append(dataset.ImageOrientationPatient)
        b_values_sort.append(dataset[0x19, 0x100c].value)
        files.append(pydicom.dcmread(fname)) 
 
    sort_index = np.argsort(b_values_sort) 
    
    b_values_sort.sort()
   
    slice_b_values= []
    slice_sorted_b_values= []

    for f in files: 
        slice_b_values.append(f)

    # sorted slices using sorted b values
    for i in range(0, len(slice_b_values)):
         slice_sorted_b_values.append(slice_b_values[sort_index[i]])
   
    return b_values, b_Vec_original, image_orientation_patient, slice_sorted_b_values 


#TODO: read v-values from gradient vec file instead.
def exp_func(b, S0, ADC):
    """ mono-exponential decay function used to perform DWI-fitting.

    Args
    ----
    b (numpy.float64): object containing b values

    Returns
    -------
    'S0' and 'ADC' (numpy.ndarray): fitted parameter apparent diffusion coefficent (mm2/sec*10^-3) per pixel.
    """
    b_v = []
    for i in b: # convert to list
        b_v.append(i)
   
    return S0*np.exp(-np.array(b_v)*ADC)


def IVIM_fitting(images_to_be_fitted, signal_model_parameters):
    """ curve fit function which returns the fit, and fitted params: S0 and ADC.

    Args
    ----
    images_to_be_fitted (numpy.ndarray): pixel value for time-series (i.e. at each b-value and for each of the 3 acquired directions) with shape [x,:]
    signal_model_parameters (list): list of b_values

    Returns
    -------
    fit (list): signal model fit per pixel
    Params (list): list consisting of fitted parameter 'S0' and 'ADC' (mm2/sec*10^-3) per pixel.
    """

    b_val = signal_model_parameters[0][0]

    lb = [0,0]
    ub = [np.inf,1]
 
    K = 10
    
    # computing strt, and end index 
    strt_idx = (len(images_to_be_fitted) // 2) - (K // 2)
    end_idx = (len(images_to_be_fitted) // 2) + (K // 2)
        
    initial_guess = [0.9*np.max(images_to_be_fitted),0.0025] 

    ## process first 10 elements
    popt_x, pcov_x = curve_fit(exp_func, xdata = b_val[:10], ydata = images_to_be_fitted[:10], p0=initial_guess, bounds=(lb,ub), method='trf', maxfev=1000000) #1000000
    ## fit middle 10 elements
    popt_y, pcov_y = curve_fit(exp_func, xdata = b_val[:10], ydata = images_to_be_fitted[strt_idx: end_idx], p0=initial_guess, bounds=(lb,ub), method='trf', maxfev=1000000) #1000000
    ## fit last 10 elements
    popt_z, pcov_z = curve_fit(exp_func, xdata = b_val[:10], ydata = images_to_be_fitted[-10:], p0=initial_guess, bounds=(lb,ub), method='trf', maxfev=1000000) #1000000

    fit_ivim_x = []
    fit_ivim_x.append(exp_func(b_val[:10], *popt_x))
    fit_x = functools.reduce(operator.iconcat, fit_ivim_x, []) 
    S0_x = popt_x[0]  
    ADC_x = popt_x[1]
  
    fit_ivim_y = []
    fit_ivim_y.append(exp_func(b_val[:10], *popt_y))
    fit_y = functools.reduce(operator.iconcat, fit_ivim_y, [])
    S0_y = popt_y[0]
    ADC_y = popt_y[1]

    fit_ivim_z = []
    fit_ivim_z.append(exp_func(b_val[:10], *popt_z))
    fit_z = functools.reduce(operator.iconcat, fit_ivim_z, [])
    S0_z = popt_z[0]
    ADC_z = popt_z[1]
 
    S0 = (S0_x + S0_y + S0_z)/3
 
    ADC = (ADC_x + ADC_y + ADC_z)/3
 
    fit = []

    fit.append(fit_x)
    fit.append(fit_y)
    fit.append(fit_z)

    fit = functools.reduce(operator.iconcat, fit, [])

    Params = [S0, ADC]

    return fit, Params


def main(images_to_be_fitted, signal_model_parameters):
    """ main function for DWI model fitting at single pixel level. 

    Args
    ----
    images_to_be_fitted (numpy.ndarray): pixel value for time-series (i.e. at b-value) with shape [x,:]
    signal_model_parameters (list): list consisting of b-values, b-vec, and image_orientation_patient


    Returns
    -------
    fit (list): signal model fit per pixel
    fitted_parameters (list): list with signal model fitted parameters 'S0' and 'ADC'.  
    """
    image_orientation_patient = signal_model_parameters[2] 

    for i in range(len(image_orientation_patient)-1):
        assert image_orientation_patient[i] == image_orientation_patient[i+1], "Error in image_orientation_patient for IVIM"
    
    fit, fitted_parameters = IVIM_fitting(images_to_be_fitted, signal_model_parameters)

    return fit, fitted_parameters

