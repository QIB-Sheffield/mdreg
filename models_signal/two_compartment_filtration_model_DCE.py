"""
@authors: Kanishka Sharma, Fotios Tagkalakis, Steven Sourbron  
DCE-MRI two-compartment filtration model fit  
2021  
"""

import numpy as np
import sys  
from scipy import integrate
np.set_printoptions(threshold=sys.maxsize)


def aif_trapz(aif, time, timepoint, Hct):  
    """ This function computes the numerical integration for the AIF first and second pass.

    Args
    ----
    timepoints (list): DCE dynamic timepoints.  
    aif (list): arterial input function.  
    timepoint (int): number of baseline acquisitions.  
    Hct (float): hematocrit.  

    Returns
    -------
    first_pass_aif_new (ndarray): first pass aif from composite trapezoidal rule.  
    second_pass_aif_new (ndarray): second pass aif from composite trapezoidal rule.  
    """

    aif0 =  np.mean(aif[0:timepoint])
    aif_new = (aif-aif0)/(1-Hct)

    first_pass_aif_new = integrate.cumtrapz(aif_new,time)
    first_pass_aif_new = np.insert(first_pass_aif_new,0,0)#add extra zero to make array back to 265
    second_pass_aif_new = integrate.cumtrapz(first_pass_aif_new,time)
    second_pass_aif_new = np.insert(second_pass_aif_new,0,0)#add extra zero to make array back to 265
    return first_pass_aif_new, second_pass_aif_new

 
def Linear_Least_Squares_2CFM(images_to_be_fitted, time, timepoint, first_pass_aif_new, second_pass_aif_new, return_parameters=True): 
    """ Linear least squares 2-compartment filtration model fit.

    Args
    ----
    images_to_be_fitted (numpy.ndarray): input image at all time-series (i.e. at each DCE dynamic measurement) with shape [x-dim*y-dim, total time-series].    
    time (list): corresponding timepoints at each AIF.  
    timepoint (int): user-defined timepoint.  
    first_pass_aif_new (ndarray): first pass aif from composite trapezoidal rule
    second_pass_aif_new (ndarray): second pass aif from composite trapezoidal rule
    return_parameters (condition): User-defined condition to return paramter maps. Default is True. If False then empty parameter maps are returned.
    
    Returns
    -------
    Sfit (numpy.ndarray): signal model fit at all time-series with shape [x-dim*y-dim, total time-series].  
    Fp (numpy.ndarray): fitted parameter 'Fp' with shape [x-dim*y-dim].  
    Tp (numpy.ndarray): fitted parameter 'Tp' with shape [x-dim*y-dim].  
    PS (numpy.ndarray): fitted parameter 'PS' with shape [x-dim*y-dim].  
    Te (numpy.ndarray): fit parameter 'Te' with shape [x-dim*y-dim].  
    """
    shape = np.shape(images_to_be_fitted) 
    S0 = np.empty(shape[0])
    St = images_to_be_fitted # signal
    Ct = np.empty(shape) #concentration
    Sfit = np.empty(shape)
    Cfit = np.empty(shape)
    
    for x in range(shape[0]):#pixels
      S0[x] = np.mean(St[x,0:timepoint]) # timepoint = 15 baselines only
      Ct[x,:] = St[x,:]-S0[x]
       
    time = np.tile(time, (shape[0],1)) # tile to repeat to match ct_new shape

    first_pass_ct_new  = integrate.cumtrapz(Ct,time)
    first_pass_ct_new = np.insert(first_pass_ct_new,0,0, axis=1)#add extra zero to make array back to 265

    second_pass_ct_new = integrate.cumtrapz(first_pass_ct_new,time)
    second_pass_ct_new = np.insert(second_pass_ct_new,0,0, axis=1)#add extra zero to make array back to 265
  
    X = np.empty([shape[0],4]) 
    A = np.empty([265,4])
    A[:,2] = second_pass_aif_new
    A[:,3] = first_pass_aif_new
    alpha = np.empty(shape[0]) 
    beta = np.empty(shape[0]) 
    gamma  = np.empty(shape[0]) 
    Fp = np.empty(shape[0]) 

    for x in range(shape[0]):
        A[:,0] = - second_pass_ct_new[x,:]
        A[:,1] = - first_pass_ct_new[x,:]
        X[x,:] = np.linalg.lstsq(A,Ct[x,:],rcond=None)[0] 
        Cfit[x,:] =X[x,0]*A[:,0] + X[x,1]*A[:,1] + X[x,2]*A[:,2] + X[x,3]*A[:,3] 
        Sfit[x,:] = S0[x]+Cfit[x,:]
        alpha[x] = X[x,0]
        beta[x]  = X[x,1]
        gamma[x]  = X[x,2]
        Fp[x]  = X[x,3]
    
    if return_parameters:
         
        if alpha.all() == 0: # TODO: conditions TBC with Steven
            Tp = 1/beta
            PS = np.zeros(shape[0]) 
            Te = np.zeros(shape[0]) 
        else:    
            if alpha.all() == 0 and beta.all() == 0: 
               Fp = np.zeros(shape[0])
               Tp = np.zeros(shape[0])
               PS = np.zeros(shape[0])
               Te = np.zeros(shape[0])
            else:   
             T = gamma/(alpha*Fp)  
             det = np.square(beta)-4*alpha
             if det < 0 :
                Tp = beta/(2*alpha)
                Te = beta/(2*alpha)
             else:
                Tp = (beta - np.sqrt(np.square(beta)-4*alpha))/(2*alpha)
                Te = (beta + np.sqrt(np.square(beta)-4*alpha))/(2*alpha)
             if Te == 0:
                Fp = np.zeros(shape[0])
                Tp = np.zeros(shape[0])
                PS = np.zeros(shape[0])
             else:    
                PS = Fp*(T-Tp)/Te      

    else:    
        Fp = np.zeros(shape[0])
        Tp = np.zeros(shape[0])
        PS = np.zeros(shape[0])   
        Te = np.zeros(shape[0])  

    return Sfit, Fp, Tp, PS, Te 



def main(images_to_be_fitted, signal_model_parameters):
    """ main function that performs 2-compartment filtration DCE model fit.  

    Args
    ----
    images_to_be_fitted (numpy.ndarray): input image at all time-series (i.e. at each DCE dynamic measurement) with shape [x-dim*y-dim, total time-series].    
    signal_model_parameters (list): list containing AIF, time, (user-defined) timepoint, (user-defined) Hematocrit as list elements.    

    Returns
    -------
    fit (numpy.ndarray): signal model fit at all time-series (i.e. at each DCE dynamic measurement) with shape [x-dim*y-dim, total time-series].   
    fitted_parameters (numpy.ndarray): output signal model fit parameters (Fp, Tp, Ps, Te) stored in a single nd-array with shape [4, x-dim*y-dim].   
    """
    
    AIF = signal_model_parameters[0]
 
    time = signal_model_parameters[1]

    timepoint = signal_model_parameters[2]
    
    Hct = signal_model_parameters[3]
  
    first_pass_aif_new, second_pass_aif_new = aif_trapz(AIF, time, timepoint, Hct)
  
    results = Linear_Least_Squares_2CFM(images_to_be_fitted, time, timepoint, first_pass_aif_new, second_pass_aif_new, return_parameters=True) 

    fit = results[0]
    Fp = results[1]
    Tp = results[2]
    Ps = results[3]
    Te = results[4]

    fitted_parameters_tuple = (Fp, Tp, Ps, Te)
    fitted_parameters = np.vstack(fitted_parameters_tuple)

    return fit, fitted_parameters



