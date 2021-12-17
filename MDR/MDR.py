"""
MODEL DRIVEN REGISTRATION (MDR) for quantitative MRI  
MDR Library for 2D image registration  
@Kanishka Sharma  
@Joao Almeida e Sousa  
@Steven Sourbron  
2021
"""

import os
import numpy as np
import SimpleITK as sitk
import itk
import copy
import pandas as pd


def model_driven_registration(images, image_parameters, model, signal_model_parameters, elastix_model_parameters, precision = 1, function = 'main'): 
    """ main function that performs the model driven registration.

    Args:
    ----
        images (numpy.ndarray): unregistered 2D images (uint16) from the selected single MR slice with shape [x-dim, y-dim, total timeseries]   
        image_parameters (sitk tuple): distance between pixels (in mm) along each dimension  
        model (module): import module for the signal model to fit  
        signal_model_parameters (list[int]): list consisting of signal model input parameters.    
        eg: TE (echo times) as input parameter (independent variable) for T2*sequence model fit  
        elastix_model_parameters (itk-elastix.ParameterObject): elastix file registration parameters  
        precision (int, optional): precision (in mm) to define the convergence criterion for MDR. Defaults to 1. Lower value means higher precision  
        function (string attribute, optional): name (user-defined) of the main model-fit function. Default is 'main'.  
    
    Returns:
    -------
        coregistered (numpy.ndarray): ffd based co-registered (uint16) 2D images as np-array with shape: [x-dim,y-dim,  total time-series] of the selected input MR slice  
        fit (numpy.ndarray): signal model fit images (uint16) of specified MR slice with shape [x-dim,y-dim,  total time-series]  
        deformation_field (numpy.ndarray): output 2D deformation fields, 'deformation_field_x' and 'deformation_field_y' as np-array with shape [x-dim, y-dim, 2,  total timeseries]  
        par (numpy.ndarray): signal model-fit output parameters  
        improvement (dataframe): maximum deformation per pixel calculated as the euclidean distance of difference between old and new deformation field appended to a dataframe until convergence criterion is met.
    """
 
    shape = np.shape(images)
    improvement = []  

    # Initialise the solution
    coregistered = copy.deepcopy(images) 
    coregistered =  np.reshape(coregistered,(shape[0]*shape[1],shape[2]))
    deformation_field = np.empty((shape[0]*shape[1],2,shape[2]))

    converged = False
    while not converged: 

        # Update the solution
        fit, par = fit_signal_model_image(coregistered, model, signal_model_parameters, function='main')
        fit = np.reshape(fit,(shape[0],shape[1],shape[2]))
        # perform 2D image registration
        coregistered, new_deformation_field = fit_coregistration(fit, images, image_parameters, elastix_model_parameters)
        # check convergence    
        improvement.append(maximum_deformation_per_pixel(deformation_field, new_deformation_field))
        converged = improvement[-1] <= precision           
        deformation_field = new_deformation_field

    coregistered = np.reshape(coregistered,(shape[0],shape[1],shape[2]))
    deformation_field = np.reshape(deformation_field,(shape[0],shape[1],2,shape[2])) 
    improvement = pd.DataFrame({'maximum_deformation_per_pixel': improvement}) 

    return coregistered, fit, deformation_field, par, improvement

# signal model function for MDR
def fit_signal_model_image(time_curve, model, signal_model_parameters, function='main'): 
    """
        This function takes signal time curve, signal model parameters, and model fit function name as input 
        and returns the fitted signal and associated output model parameters.
    """
    fit, fitted_parameters = getattr(model, function)(time_curve, signal_model_parameters)
    
    return fit, fitted_parameters


def fit_coregistration(fit, images, image_parameters, elastix_model_parameters):
    """Co-register the 2D fit-image with the unregistered 2D input image.

    Args:
    ----
    fit (numpy.ndarray): signal model fit images (single 2D slice with all time-series) with shape: [x-dim,y-dim, total timeseries]  
    images (numpy.ndarray): unregistered 2D images (uint16, single 2D slice with all time-series) as np-array with shape [x-dim,y-dim, total timeseries]  
    image_parameters (sitk tuple): distance between pixels (in mm) along each dimension  
    elastix_model_parameters (itk-elastix.ParameterObject): elastix file registration parameters  

    Returns:
    -------
    coregistered (numpy.ndarray): coregisterd 2D images with shape [x-dim * y-dim, total time-series]  
    deformation_field (numpy.ndarray): output 2D deformation fields with shape [x-dim, y-dim, 2, num of dynamics].  
    Dimension '2' corresponds to deformation_field_x and deformation_field_y.  
    """
    
    shape = np.shape(images)
    coregistered = np.empty((shape[0]*shape[1],shape[2]))
    deformation_field = np.empty([shape[0]*shape[1], 2, shape[2]])
    for t in range(shape[2]): #dynamics
      coregistered[:,t], deformation_field[:,:,t] = itkElastix_MDR_coregistration(images[:,:,t], fit[:,:,t], elastix_model_parameters, image_parameters)
    return coregistered, deformation_field


def maximum_deformation_per_pixel(deformation_field, new_deformation_field):
    """This function calculates diagnostics from the registration process.

    It takes as input the original deformation field and the new deformation field
    and returns the maximum deformation per pixel (in mm).
    The maximum deformation per pixel is calculated as 
    the euclidean distance of difference between the old and new deformation field. 
    """

    df_difference = deformation_field - new_deformation_field
    df_difference_x_squared = np.square(df_difference[:,0,:].squeeze())
    df_difference_y_squared = np.square(df_difference[:,1,:].squeeze())
    dist = np.sqrt(np.add(df_difference_x_squared, df_difference_y_squared))
    maximum_deformation_per_pixel = np.nanmax(dist)
    
    return maximum_deformation_per_pixel 


# deformable registration for MDR
def itkElastix_MDR_coregistration(target, source, elastix_model_parameters, image_parameters):
    """
        This function takes pair-wise unregistered source image and target image (per time-series point) as input 
        and returns ffd based co-registered target image and its corresponding deformation field. 
    """
    shape_source = np.shape(source)
    shape_target = np.shape(target)

    source = sitk.GetImageFromArray(source)
    source.SetSpacing(image_parameters)
    source.__SetPixelAsUInt16__
    source = np.reshape(source, [shape_source[0], shape_source[1]]) 
    
    target = sitk.GetImageFromArray(target)
    target.SetSpacing(image_parameters)
    target.__SetPixelAsUInt16__
    target = np.reshape(target, [shape_target[0], shape_target[1]])
    
    ## read the source and target images
    elastixImageFilter = itk.ElastixRegistrationMethod.New()
    elastixImageFilter.SetFixedImage(itk.GetImageFromArray(np.array(source, np.float32)))
    elastixImageFilter.SetMovingImage(itk.GetImageFromArray(np.array(target, np.float32)))

    ## call the parameter map file specifying the registration parameters
    elastixImageFilter.SetParameterObject(elastix_model_parameters) 
    ## print Parameter Map
    print(elastix_model_parameters)

    ## set additional options
    elastixImageFilter.SetNumberOfThreads(os.cpu_count()-1)
    elastixImageFilter.SetLogToConsole(True)
    ## update filter object (required)
    elastixImageFilter.UpdateLargestPossibleRegion()

    ## RUN ELASTIX using ITK-Elastix filters
    coregistered = itk.GetArrayFromImage(elastixImageFilter.GetOutput()).flatten()

    transformixImageFilter = itk.TransformixFilter.New()
    transformixImageFilter.SetTransformParameterObject(elastixImageFilter.GetTransformParameterObject())
    transformixImageFilter.ComputeDeformationFieldOn()
    transformixImageFilter.SetMovingImage(itk.GetImageFromArray(np.array(target, np.float32)))
    deformation_field = itk.GetArrayFromImage(transformixImageFilter.GetOutputDeformationField()).flatten()
    deformation_field = np.reshape(deformation_field, [int(len(deformation_field)/2), 2])

    return coregistered, deformation_field

