"""
MDR main script: @Kanishka Sharma 
"""

import os
import sys
import numpy as np
import subprocess
import SimpleITK as sitk
sys.path.insert(1, 'C:\\Users\\medkshaa\\Documents\\GitHub')
from MDR_Library.models.iBEAt.DCE import iBEAt_DCE


def signal_model_fit(time_curve, signal_model_parameters): 

    fitted_parameters = getattr(iBEAt_DCE, signal_model_parameters[0])(time_curve, signal_model_parameters)
    
    return fitted_parameters
  
#dodo ks: remove output path from here: check elastix commandline tool; 
# elastix accepts cmdline with filenames and paths; 
#elastix does not accept numpy array for input - only original file paths accpeted
def elastix_ffd_coregistration(source, target, elastix_parameter_file, output_dir):
   
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    print("to commandline")    
    cmd = [ 'elastix', '-m', target, '-f', source, '-out', output_dir, '-p', elastix_parameter_file]
    try:
        subprocess.check_call(cmd)
    except:
        print ('Image registration failed')
        print (sys.exc_info())        
    
    
  
def elastix_MDR(images, signal_model_parameters, elastix_parameter_file, output_dir, stopping_criterion = None):
  
  if stopping_criterion is None:
    stopping_criterion = 0.1 
  
  
  coregistered = images

  fit = images

  #test
  fitted = signal_model_fit(coregistered, signal_model_parameters) 

  #test
  elastix_ffd_coregistration(images, fit, elastix_parameter_file, output_dir)

 # shape = np.shape(images)
 # for x in range(shape[0]*shape[1]):
  # fit[x,:] = signal_model_fit(coregistered[x,:], signal_model_parameters)
  # for t in range(shape[2]):#dynamics
  #   coregistered[:,t], deformation_fields[:,t] = elastix_ffd_coregistration(images[:,t], fit[:,t], elastix_parameter_file, output_dir)
  # converged = False
 
  # while converged:
  #     for x in range(shape[0]*shape[1]):#PIXELS
  #       fit[x,:]= signal_model_fit(coregistered[x,:], signal_model_parameters)
  #     for t in range(shape[2]):#dynamics
  #       coregistered[:,t], new_deformation_fields[:,t] = elastix_ffd_coregistration(images[:,t], fit[:,t], elastix_parameter_file)
  #     converged = max(abs(deformation_fields - new_deformation_fields)) < stopping_criterion
  #     deformation_field = new_deformation_field
      
  return #coregistered, deformation_field
  

#to create deformation field and apply on full image
def transformix_deformation_field(output_dir, transform_parameters_file):
    cmd = [ 'transformix', '-def', 'all', '-out', output_dir, '-tp', transform_parameters_file]
    try:
        subprocess.check_call(cmd)
    except:
        print ('Transformix failed')
        print (sys.exc_info())
        
     




