"""
MODEL DRIVEN REGISTRATION (MDR) for quantitative renal MRI
MDR Library
@Kanishka Sharma 
@Steven Sourbron
2021
"""

import os
import numpy as np
import SimpleITK as sitk
import itk
import pandas as pd

def model_driven_registration(images, image_parameters, signal_model_parameters, elastix_model_parameters, precision = 1): 
    """ main function that performs the model driven registration.

    Args:
    ----
        images (numpy.ndarray): unregistered 2D images (uint16) with shape [x-dim,y-dim, number of slices]
        image_parameters (sitk tuple): distance between pixels (in mm) along each dimension
        signal_model_parameters (list): a list consisting of a constant 'MODEL' which is the imported signal model and signal model specific parameters as the subsequent elements of the list.
        eg - [MODEL, T2_prep_times], where MODEL is the python script within the 'model' module containing the T2 signal model and 'T2_prep_times' are the T2 specific model input parameters. 
        elastix_model_parameters (SimpleITK.ParameterMap): elastix file registration parameters
        precision (int, optional): precision (in mm) to define the convergence criterion for MDR. Defaults to 1.

    Returns:
    -------
        coregistered (numpy.ndarray): ffd based co-registered (uint16) 2D images as np-array with shape: [x-dim,y-dim, number of slices]
        fit (numpy.ndarray: signal model fit image (uint16) as np-array and shape [x-dim,y-dim, num of slices]
        deformation_field (numpy.ndarray): output deformation fields - deformation_field_x and deformation_field_y as np-array with shape [x-dim, y-dim, 2, num of slices]
        par (list): signal model-fit parameters as a list
        improvement (dataframe): maximum deformation per pixel calculated as the euclidean distance of difference between old and new deformation field appended to a dataframe until convergence criterion is met.
    
    """
    shape = np.shape(images)
    improvement = []  

    # Initialise the solution
    coregistered =  np.reshape(images,(shape[0]*shape[1],shape[2]))
    deformation_field = np.zeros((shape[0]*shape[1],2,shape[2]))

    converged = False
    while not converged: 

        # Update the solution
        fit, par = fit_signal_model_image(shape, coregistered, signal_model_parameters)
        coregistered, new_deformation_field = fit_coregistration(shape, fit, images, image_parameters, elastix_model_parameters)

        # check convergence    
        improvement.append(maximum_deformation_per_pixel(deformation_field, new_deformation_field))
        converged = improvement[-1] <= precision           
        deformation_field = new_deformation_field

    coregistered = np.reshape(coregistered,(shape[0],shape[1],shape[2]))
    deformation_field = np.reshape(deformation_field,(shape[0],shape[1],2,shape[2])) 
    improvement = pd.DataFrame({'maximum_deformation_per_pixel': improvement}) 

    return coregistered, fit, deformation_field, par, improvement


def fit_signal_model_image(shape, coregistered, signal_model_parameters):
    """Fit signal model images.
    
    Args:
    ----
    shape (tuple): tuple with original image shape [x-dim,y-dim,z-dim]
    coregistered (numpy.ndarray): co-registerd 2D images as np-array with shape [x-dim * y-dim, number of slices]
    signal_model_parameters (list): a list consisting of a constant 'MODEL' which is the imported signal model and signal model specific parameters as the subsequent elements of the list.
    eg - [MODEL, T2_prep_times], where MODEL is the python script within the 'model' module containing the T2 signal model and 'T2_prep_times' are the T2 specific model input parameters. 
    
    Returns:
    -------
    fit (numpy.ndarray): signal model fit images (2D slices) as np-array with shape: [x-dim,y-dim, num of slices]
    par (list): signal model-fit parameters as a list.
    """

    fit = np.zeros((shape[0]*shape[1],shape[2]))
    par = np.array([]) 
    for x in range(shape[0]*shape[1]):#pixels
      fit[x,:], par_x = fit_signal_model_pixel(coregistered[x,:], signal_model_parameters) 
      par = np.append(par, par_x)
    fit = np.reshape(fit,(shape[0],shape[1],shape[2]))
    return fit, par


def fit_coregistration(shape, fit, images, image_parameters, elastix_model_parameters):
    """Co-register the 2D fit-image with the unregistered 2D input image.

    Args:
    ----
    shape (tuple): tuple with original image shape [x-dim,y-dim,z-dim]
    fit (numpy.ndarray): signal model fit images (2D slices) with shape: [x-dim,y-dim, num of slices]
    images (numpy.ndarray): unregistered 2D images (uint16) as np-array with shape [x-dim,y-dim, number of slices]
    image_parameters (sitk tuple): image parameters define the pixel spacing in the image
    elastix_model_parameters (list): elastix parameter file parameters

    Returns:
    -------
    coregistered (numpy.ndarray: coregisterd 2D images with shape [x-dim * y-dim, number of slices]
    deformation_field (numpy.ndarray): output deformation fields with shape [x-dim, y-dim, 2, num of slices]. Dimension '2' corresponds to deformation_field_x and deformation_field_y. 
    """

    coregistered = np.zeros((shape[0]*shape[1],shape[2]))
    deformation_field = np.zeros([shape[0]*shape[1], 2, shape[2]])
    for t in range(shape[2]): #dynamics
      coregistered[:,t], deformation_field[:,:,t] = itkElastix_MDR_coregistration(images[:,:,t], fit[:,:,t], elastix_model_parameters, image_parameters)
    return coregistered, deformation_field


def maximum_deformation_per_pixel(deformation_field, new_deformation_field):
    """
    This function calculates diagnostics from the registration process
    It takes as input the original deformation field and the new deformation field
    and returns the maximum deformation per pixel.
    The maximum deformation per pixel calculated as 
    the euclidean distance of difference between old and new deformation field. 
    """

    df_difference = deformation_field - new_deformation_field
    df_difference_x_squared = np.square(df_difference[:,0,:].squeeze())
    df_difference_y_squared = np.square(df_difference[:,1,:].squeeze())
    dist = np.sqrt(np.add(df_difference_x_squared, df_difference_y_squared))
    maximum_deformation_per_pixel = np.nanmax(dist)
    
    return maximum_deformation_per_pixel 

# signal model function for MDR
def fit_signal_model_pixel(time_curve, signal_model_parameters): 
    """
        This function takes signal time curve and signal model parameters as input 
        and returns the fitted signal and associated model parameters.
    """
    fit, fitted_parameters = getattr(signal_model_parameters[0], 'main')(time_curve, signal_model_parameters[1])
    return fit, fitted_parameters


# deformable registration for MDR
def itkElastix_MDR_coregistration(target, source, elastix_model_parameters, image_parameters):
    """
        This function takes unregistered source image and target image as input 
        and returns ffd based co-registered image and corresponding deformation field. 
    """
    shape_source = np.shape(source)
    shape_target = np.shape(target)

    source = sitk.GetImageFromArray(source)
    #source.SetOrigin(image_parameters[0]) # not required for MDR based time-series registration only
    source.SetSpacing(image_parameters)
    source.__SetPixelAsUInt16__
    source = np.reshape(source, [shape_source[0], shape_source[1]]) 
    
    target = sitk.GetImageFromArray(target)
    #target.SetOrigin(image_parameters[0]) # not required for MDR based time-series registration only
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

