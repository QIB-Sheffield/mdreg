#!/ur/bin/python
# -*- coding: utf-8 -*-
"""
@author: Steven Sourbron  
T2-mapping signal model-fit  
2022 
"""
import numpy as np
from .exp_decay import main as exp_decay

def pars():
    return ['S0', 'T2']

def bounds():
    lower = [0,0]
    upper = [np.inf, 250]
    return lower, upper

def main(images, T2_prep_times):
    """ main function that performs the T2*-map signal model-fit for input 2D image at multiple time-points (TEs).

    Args
    ----
    images_to_be_fitted (numpy.ndarray): input image at all time-series (i.e. at each T2_prep_time) with shape [x-dim*y-dim, total time-series].  
    signal_model_parameters (list): list containing TE times as list elements.  

    Returns
    -------
    fit (numpy.ndarray): signal model fit per pixel for whole image with shape [x-dim*y-dim, total time-series].  
    fitted_parameters (numpy.ndarray): output signal model fit parameters 'S0' and 'T2star' stored in a single nd-array with shape [2, x-dim*y-dim].      
    """
    fit, par = exp_decay(
        images, T2_prep_times, 
        lower_bounds = [0, 0.001],
        initial_value = [1.0, 1.0/80], 
        maxfev = 500, 
    )
    par[:,1] = 1/par[:,1]

    return fit, par
