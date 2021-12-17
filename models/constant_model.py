#!/ur/bin/python
# -*- coding: utf-8 -*-
"""
@author: Kanishka Sharma  
constant model for MDR on MT sequence, etc.  
2021  
"""

import numpy as np

def constant_model_fit(images_to_be_fitted, return_parameter='MTR'):
    """ constant_model_fit function for MT sequence.  

    Args
    ----
    images_to_be_fitted (numpy.ndarray): input image at all time-series (eg: MT_OFF, MT_ON pair) with shape [x-dim*y-dim, total time-series].      
    return_parameter (string): user-specified string to return parameter map. Default is 'MTR' for MT maps. If none specified then empty maps generated.  

    Returns
    -------
    fit (numpy.ndarray): signal model fit at all time-series with shape [x-dim*y-dim, total time-series].    
    fitted_params (numpy.ndarray): default is the fitted parameter 'MTR' with shape [x-dim*y-dim].    
    """
    
    shape = np.shape(images_to_be_fitted)
    fit = np.empty(shape)
    model_param = np.mean(images_to_be_fitted, axis=1)
   
    for i in range(shape[1]):#timepoints
        fit[:,i] =  model_param
       
    if return_parameter=='MTR':
       MTR = 100*((images_to_be_fitted[:,0]-images_to_be_fitted[:,1])/images_to_be_fitted[:,0])
       fitted_params = MTR
    else: # TODO future version: if sequence is MT then return MTR else fitted params=0
        fitted_params = np.zeros(shape[0])
 
    return fit, fitted_params



def main(images_to_be_fitted, signal_model_parameters): 
    """ main function that performs a constant model-fit for input 2D image at multiple time-points. 

    Args
    ----
    images_to_be_fitted (numpy.ndarray): input image at all time-series (eg: MT-OFF and MT-ON pair) with shape [x-dim*y-dim, total time-series].    
    signal_model_parameters (list): list containing independent variable elements.   

    Returns
    -------
    fit (numpy.ndarray): signal model fit at all time-series (eg: MT_OFF, MT_ON fits) with shape [x-dim*y-dim, total time-series].   
    fitted_parameters (numpy.ndarray): output signal model fit parameter 'MTR' stored in a single nd-array with shape [2, x-dim*y-dim].     
    """

    independent_variable = signal_model_parameters # signal model parameter is initialised to zero for MT sequence

    results = constant_model_fit(images_to_be_fitted, return_parameter='MTR')
 
    fit = results[0]
    fitted_parameters_tuple = results[1]
    fitted_parameters = np.array(fitted_parameters_tuple)
    
    return fit, fitted_parameters




