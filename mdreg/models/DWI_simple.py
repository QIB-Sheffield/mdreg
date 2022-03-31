"""
@author: Steven Sourbron 
DWI monoexponential model fit for the MDR Library  
2022 
"""

import numpy as np 
from .exp_decay import main as exp_decay

def pars():
    return ['S0', 'ADC']

def bounds():
    lower = [0,0]
    upper = [np.inf, 1.0]
    return lower, upper

def main(images, bvalues):
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
    
    initial_value = [0.9, 0.0025]
    maxfev = 500

    shape = np.shape(images)
    par = np.empty((3, shape[0], 2))
    fit = np.empty(shape)
    nb = len(bvalues)
    
    fit[:,:nb], par[0,:,:] = exp_decay(
        images[:,:nb], bvalues, 
        initial_value = initial_value, 
        maxfev = maxfev, 
    )
    fit[:,nb:2*nb], par[1,:,:] = exp_decay(
        images[:,nb:2*nb], bvalues, 
        initial_value = initial_value, 
        maxfev = maxfev, 
    )
    fit[:,2*nb:], par[2,:,:] = exp_decay(
        images[:,2*nb:], bvalues, 
        initial_value = initial_value, 
        maxfev = maxfev, 
    )
   
    return fit, np.mean(par, axis=0)

