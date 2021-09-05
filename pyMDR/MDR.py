"""
MODEL DRIVEN REGISTRATION for quantitative renal MRI
MDR Library
@Kanishka Sharma 
@Steven Sourbron
2021
"""

import numpy as np
import SimpleITK as sitk
import time
import pandas as pd


## 
def calculate_diagnostics(deformation_field, new_deformation_field):
    """
    This function calculates diagnostics from the registration process
    it takes as input the original deformation field and the new deformation field
    and returns maximum deformation per pixel
    """

    df_difference = deformation_field - new_deformation_field

    df_difference_x_squared = np.square(df_difference[:,0,:].squeeze())
    df_difference_y_squared = np.square(df_difference[:,1,:].squeeze())

    dist = np.sqrt(np.add(df_difference_x_squared, df_difference_y_squared))

    maximum_deformation_per_pixel = np.nanmax(dist)

    return maximum_deformation_per_pixel



# signal model function for MDR
def signal_model_fit(time_curve, signal_model_parameters): 
    """
        This function takes signal time curve and signal model paramters as input 
        and returns the fitted signal and model parameters
    """

    fit, fitted_parameters = getattr(signal_model_parameters[0][0], signal_model_parameters[0][1])(time_curve, signal_model_parameters) 
  
    return fit, fitted_parameters



# deformable registration for MDR
def simpleElastix_ffd_coregistration(target, source, elastix_model_parameters, slice_parameters):
    """
        This function takes source image and target image as input 
        and returns ffd based co-registered image and deformation field 
    """
    shape_source = np.shape(source)
    shape_target = np.shape(target)

    ## TODO for 3D; OK for 2D images
    source = sitk.GetImageFromArray(source)
    source.SetOrigin(slice_parameters[0])
    source.SetSpacing(slice_parameters[1])
    source.__SetPixelAsUInt16__
    source = np.reshape(source, [shape_source[0], shape_source[1]]) 
    
    target = sitk.GetImageFromArray(target)
    target.SetOrigin(slice_parameters[0])
    target.SetSpacing(slice_parameters[1])
    target.__SetPixelAsUInt16__
    target = np.reshape(target, [shape_target[0], shape_target[1]])
    
    ## read the source and target images
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(sitk.GetImageFromArray(source))
    elastixImageFilter.SetMovingImage(sitk.GetImageFromArray(target))

    ## call the parameter map file specifying the registration parameters
    elastixImageFilter.SetParameterMap(elastix_model_parameters) 
    elastixImageFilter.PrintParameterMap()

    ## RUN ELASTIX using SimpleITK filters
    elastixImageFilter.Execute()
    coregistered = elastixImageFilter.GetResultImage()

    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
    transformixImageFilter.ComputeDeformationFieldOn()
    transformixImageFilter.Execute()
    deformation_field = transformixImageFilter.GetDeformationField()

    return coregistered, deformation_field

## TODO: change the function call with modified variable names
#def model_driven_registration(images, image_parameters, signal_model_parameters, elastix_model_parameters, precision = 1): # precision is in mm
def model_driven_registration(original_images, slice_parameters, signal_model_parameters, elastix_model_parameters, precision = 1): # precision is in mm
    """
    This is the main function to call Model Driven Registration 
    images: the unregistered images as nd-array
    image_parameters: [image origin, image spacing]
    signal_model_parameters: [MODEL, model specific parameters]
    elastix_model_parameters:
    and returns ffd based co-registered image, fitted image, deformation field, fitted parameters and diagnostics 
  """
    shape = np.shape(original_images)
    
    coregistered =  np.zeros([shape[0]*shape[1], shape[2]])
    coregistered =  np.reshape(original_images,(shape[0]*shape[1],shape[2]))
   
    deformation_field = np.zeros([shape[0]*shape[1], 2, shape[2]]) # todo for 3D
    new_deformation_field = np.zeros([shape[0]*shape[1], 2, shape[2]]) # todo for 3d
    
    # create a list for diagnostics
    improvement = []  
    fit = np.zeros([shape[0]*shape[1], shape[2]])
    
    ## store Parameters generated from fits
    Par = np.array([])
    
    start_computation_time = time.time()

    for x in range(shape[0]*shape[1]): #pixels
      
      fit[x,:], Par_x  = signal_model_fit(coregistered[x,:], signal_model_parameters)
      Par = np.append(Par, Par_x)

    fit = np.reshape(fit,(shape[0],shape[1],shape[2]))
    
    for t in range(shape[2]): #dynamics

      coregistered[:,t], deformation_field[:,:,t] = simpleElastix_ffd_coregistration(original_images[:,:,t], fit[:,:,t], elastix_model_parameters, slice_parameters)
      
    converged = False


    while not converged: 

        fit = np.reshape(fit,(shape[0]*shape[1],shape[2]))
        Par = np.array([]) 

        for x in range(shape[0]*shape[1]):#pixels
          fit[x,:], Par_x = signal_model_fit(coregistered[x,:], signal_model_parameters) 
          Par = np.append(Par, Par_x)
        
        fit = np.reshape(fit,(shape[0],shape[1],shape[2]))
        
        for t in range(shape[2]):#dynamics
          coregistered[:,t], new_deformation_field[:,:,t] = simpleElastix_ffd_coregistration(original_images[:,:,t], fit[:,:,t], elastix_model_parameters, slice_parameters)

           
        ## calculate diagnostics: maximum_deformation_per_pixel  
        maximum_deformation_per_pixel = calculate_diagnostics(deformation_field, new_deformation_field)        
        print("maximum_deformation_per_pixel")
        print(maximum_deformation_per_pixel)

        improvement.append(maximum_deformation_per_pixel)
        # TODO: correct bug: only stores 1st and last max. deformation
        diagnostics_dict = {'maximum_deformation_per_pixel': improvement}
        
        # update the deformation field
        deformation_field = new_deformation_field

        if maximum_deformation_per_pixel <= precision: # elastix for physical units (in mm)
            print("MDR converged! final improvement = " + str(maximum_deformation_per_pixel))
            converged = True
    
    end_computation_time = time.time()
    print("total computation time for MDR (minutes taken:)...")
    print(0.0166667*(end_computation_time - start_computation_time)) # in minutes
    print("completed MDR registration!")

    diagnostics = pd.DataFrame(diagnostics_dict) 

    coregistered = np.reshape(coregistered,(shape[0],shape[1],shape[2]))
    deformation_field = np.reshape(deformation_field,(shape[0],shape[1],2,shape[2]))

    MDR_output = []

    MDR_output = [coregistered, fit, deformation_field, Par, diagnostics] 

    return MDR_output
   