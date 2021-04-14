"""
@KanishkaS - re-written from @Fotios Tagkalakis
"""

from tqdm import tqdm
import numpy as np
import sys  
import time
from numpy import trapz
from scipy.signal import argrelextrema
import multiprocessing as mp
np.set_printoptions(threshold=sys.maxsize)

def load_txt(full_path_txt):
    counter_file = open(full_path_txt, 'r+')
    content_lines = []
    for cnt, line in enumerate(counter_file):
        content_lines.append(line)
    x_values_index = content_lines.index('X-values\n')
    assert (content_lines[x_values_index+1]=='\n')
    y_values_index = content_lines.index('Y-values\n')
    assert (content_lines[y_values_index+1]=='\n')
    time = list(map(lambda x: float(x), content_lines[x_values_index+2 : y_values_index-1]))
    aif = list(map(lambda x: float(x), content_lines[y_values_index+2 :]))
    return aif, time

def Integral_Trapezoidal_Rule_initial(x,time):
    first_pass=[]
    for tm in range(len(time)):
        first_pass.append(trapz(x[0:tm+1],time[0:tm+1]))
    return first_pass

def Integral_Trapezoidal_Rule_second(first_pass,time):
    second_pass=[]
    for tm in range(len(time)):
        second_pass.append(trapz(first_pass[0:tm+1],time[0:tm+1]))
    return second_pass

def Linear_Least_Squares_2CFM(ct, timepoints, aif, timepoint = 39,Hct = 0.45):
  
    time =  timepoints
   
    ct0 = np.mean(ct[0:timepoint])
    aif0 =  np.mean(aif[0:timepoint])
    ct_new = ct-ct0

    aif_new = (aif-aif0)/(1-Hct)

    #initialization of matrix A
    A=np.array([0,0,0,0]) 

    first_pass_ct_new=Integral_Trapezoidal_Rule_initial(ct_new,time)
    first_pass_aif_new=Integral_Trapezoidal_Rule_initial(aif_new,time)
    second_pass_ct_new=Integral_Trapezoidal_Rule_second(first_pass_ct_new,time)
    second_pass_aif_new=Integral_Trapezoidal_Rule_second(first_pass_aif_new,time)
    for t in range(0,len(time)):
        A1_1=second_pass_ct_new[t]
        A1_2=first_pass_ct_new[t]
        A1_3=second_pass_aif_new[t]
        A1_4=first_pass_aif_new[t]
        A_next=np.array([-A1_1,-A1_2,A1_3,A1_4])
        A=np.vstack((A,A_next))    

    # Drop first row of matrix [A] which is full of zeros and was used for initialization purposes
    A=np.delete(A,(0),axis=0)

    X = np.linalg.lstsq(A,ct_new,rcond=None)[0]

    # Extract physical parameters
    alpha = X[0]
    beta = X[1]
    gamma = X[2]
    Fp = X[3]
    
    if alpha == 0 or Fp == 0 :
        Tp = 0
        PS = 0
        Te = 0
        fit = 0
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
            Fp = 0
            Tp = 0
            PS = 0
            fit = 0
        else:    
            PS = Fp*(T-Tp)/Te                       
            #params = [Fp, Tp, PS, Te]
            fit =X[0]*A[:,0] + X[1]*A[:,1] + X[2]*A[:,2] + X[3]*A[:,3]
            fit=ct0+fit
    
    BF = Fp/(1 -Hct)
  
    
    if isinstance(fit, int):
        fit = np.zeros_like(BF) # REDO TO SHAPE OF BF and remove BF
    

    return fit, Fp, Tp, PS, Te 

def DCE_find_timepoint(aif):
    # find timepoint
    idx = np.argmax(aif)
    minima = argrelextrema(np.asarray(aif), np.less)[0]
    maxima = argrelextrema(np.asarray(aif), np.greater)[0]
    assert idx in maxima, "The global maximum was not found in the maxima list"
    wanted_minimum = np.nan
    for ix, val in enumerate(minima):
        if ix==len(minima)-1:
            continue
        elif val<idx and minima[ix+1]>idx:
            wanted_minimum = val
    assert wanted_minimum + 1 < idx, "Minimum + 1 position equals to Global Maximum"
    return wanted_minimum + 1


def parallel_DCE_Fitting(images_to_be_fitted, signal_model_parameters):# images , AIF, timepoints as inputs # remove self
    """
    shape: x=192,y=192,z=265 example
    images: numpy array 3D (2D + time)
  
    """
    shape = np.shape(images_to_be_fitted)

    AIF = signal_model_parameters[1]

    timepoints = signal_model_parameters[2]

    max_timepoint = DCE_find_timepoint(AIF)


    indices = range(shape[0]*shape[1])
    images = images_to_be_fitted.reshape((shape[0]*shape[1], shape[2]))
    print(np.shape(images))
    # construct pool
    pool = mp.Pool(mp.cpu_count()) # parellel processing
    print('Parallel Processing for DCE fitting ...' )
  
    Fp = np.zeros((shape[0]*shape[1],))
    Tp = np.zeros((shape[0]*shape[1],)) 
    PS = np.zeros((shape[0]*shape[1],))
    Te = np.zeros((shape[0]*shape[1],))
    fitted = np.zeros((shape[0]*shape[1], shape[2]))

    args = [(images[idx, :], timepoints, AIF, max_timepoint) for idx in indices] # ks (ct,timepoints, aif, max_timepoint)
 

    results = pool.starmap(Linear_Least_Squares_2CFM, args)     # parallel pool   

    for index, result in zip([idx for idx in tqdm(range(len(indices)))], results):
        if np.shape(result[1])==(shape[2],):
            fitted[index,:] = result[1] 
        # For the cases where Linear_Least_Squares_2CFM return fit=0, fit gets converted to a vector full of 0
        elif np.shape(result[1])==():
            fitted[index,:] = np.zeros((shape[2]))
        Fp[index] = result[2]
        Tp[index] = result[3]
        PS[index] = result[4]
        Te[index] = result[5]

    fitted = np.array(fitted)
    fitted = fitted.reshape(shape)
    
    pool.close() 
    
    return fitted, [Fp.reshape((shape[0],shape[1])), Tp.reshape((shape[0],shape[1])), PS.reshape((shape[0],shape[1])), Te.reshape((shape[0],shape[1]))]
    
    