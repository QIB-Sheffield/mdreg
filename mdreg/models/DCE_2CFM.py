"""
@authors: Kanishka Sharma, Fotios Tagkalakis, Steven Sourbron  
DCE-MRI two-compartment filtration model fit  
2021  
"""

import numpy as np 
from scipy import integrate

def pars():
    return ['FP', 'TP', 'PS', 'TE']


def bounds():
    lower = [0,0,0,0]
    upper = [0.1, 20, 0.01, 120]
    return lower, upper


def ddint(c, t):

    ci = integrate.cumtrapz(c, t)
    ci = np.insert(ci, 0, 0)
    cii = integrate.cumtrapz(ci, t)
    cii = np.insert(cii, 0, 0)

    return cii, ci


def DCEparameters(X):

    alpha = X[0]
    beta = X[1]
    gamma = X[2]
    Fp = X[3]
    
    if alpha == 0: 
        if beta == 0:
            return [Fp, 0, 0, 0]
        else:
            return [Fp, 1/beta, 0, 0]

    nom = 2*alpha
    det = beta**2 - 4*alpha
    if det < 0 :
        Tp = beta/nom
        Te = Tp
    else:
        root = np.sqrt(det)
        Tp = (beta - root)/nom
        Te = (beta + root)/nom

    if Te == 0:
        PS = 0
    else:   
        if Fp == 0:
            PS == 0
        else:
            T = gamma/(alpha*Fp) 
            PS = Fp*(T-Tp)/Te   

    return [Fp, Tp, PS, Te] 


def main(St, p):
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

    shape = np.shape(St)
    fit = np.empty(shape)
    par = np.empty((shape[0], 4))
    
    Sa = p[0]
    t = p[1]
    baseline = p[2]
    Hct = p[3]

    S0 = np.mean(St[:,:baseline], axis=1)
    Sa0 = np.mean(Sa[:baseline])
    ca = (Sa-Sa0)/(1-Hct)
    
    A = np.empty((shape[1],4))
    A[:,2], A[:,3] = ddint(ca, t)
    for x in range(shape[0]):
        c = np.squeeze(St[x,:]) - S0[x]
        ctii, cti = ddint(c, t)
        A[:,0] = -ctii
        A[:,1] = -cti
        P = np.linalg.lstsq(A, c, rcond=None)[0] 
        fit[x,:] = S0[x] + P[0]*A[:,0] + P[1]*A[:,1] + P[2]*A[:,2] + P[3]*A[:,3]
        par[x,:] = DCEparameters(P)

    return fit, par