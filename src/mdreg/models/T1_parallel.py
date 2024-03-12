import numpy as np
import os
import sys  
from scipy.optimize import curve_fit
import multiprocessing
np.set_printoptions(threshold=sys.maxsize)

try: 
    num_workers = int(len(os.sched_getaffinity(0)))
except: 
    num_workers = int(os.cpu_count())


def pars():
    return ['a','b','T1_apparent','T1']


def bounds():
    lower = [0.0, 0.0, 0, 0] 
    upper = [np.inf, np.inf, 3000, 3000]
    return lower, upper


def signal(TI, a, b, T1app):
    return np.abs(a - b * np.exp(-TI/T1app)) 


def fit_signal(args):
    TI, sig = args
    s0 = np.amax(sig)
    pars = [s0, s0*1.9, 1500*(1.9-1)]
    try:
        pars, _ = curve_fit(signal, TI, sig, 
            p0 = pars, 
            #bounds = ([0,0,0], [np.inf, np.inf, 2000]), 
            xtol = 1e-3,
            #maxfev = 10000,
        )
    except:
        pass

    return pars


def main(images, inversion_times):
    """ Calls T1_fitting_pixel which contains the curve_fit function and returns the fit, the fitted parameters A,B, and T1.  

    Args
    ----
    images (numpy.ndarray): input image at all time-series (i.e. at each TI time) with shape [x-dim*y-dim, total time-series].    
    inversion_times (list): list containing T1 inversion times as input (independent variable) for the signal model fit.    

    Returns
    -------
    fit (numpy.ndarray): signal model fit at all time-series with shape [x-dim*y-dim, total time-series].
    T1_estimated, T1_apparent, b, a (numpy.ndarray): fitted parameters each with shape [x-dim*y-dim].    
    """
    TI = np.array(inversion_times)
    shape = np.shape(images)
    
    # Perform the fit
    pool = multiprocessing.Pool(processes=num_workers)
    args = [(TI, images[x,:]) for x in range(shape[0])]
    fit_pars = pool.map(fit_signal, args)
    pool.close()
    pool.join()

    # Create output arrays
    fit = np.empty(shape)
    pars = np.zeros((shape[0],4))
    for x, p in enumerate(fit_pars):
        fit[x,:] = signal(TI, p[0], p[1], p[2])
        pars[x,:3] = p
        if p[0] != 0:
            # T1 = T1app*(b/a-1)
            T1 = p[2]*(p[1]/p[0]-1)
            pars[x,3] = min([T1,3000])
       
    return fit, pars