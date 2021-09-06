"""
MODEL DRIVEN REGISTRATION for quantitative renal MRI
MDR Library
@Kanishka Sharma 
@Steven Sourbron
2021
"""

import numpy as np
import SimpleITK as sitk
import pandas as pd

def model_driven_registration(images, image_parameters, signal_model_parameters, elastix_model_parameters, precision = 1): # precision is in mm (TO DOCSTRING)
    """ Performs model driven registration

    Parameters
    ---------- 
    images: the unregistered images as nd-array (KANISHKA SPECIFY dimensions and date type)
    image_parameters: [image origin, image spacing] (KANISHKA SPECIFY; origin is a set of coordinates??? what si image spacing? What format is this?)
    signal_model_parameters: [MODEL, model specific parameters]
    elastix_model_parameters: elastix file registration parameters
    precision: KANISHKA??

    Return values
    ------------
    coregistered: ffd based co-registered image (TYPE, DIMENSIONS??)
    fit: fitted image
    deformation field: ???
    par: fitted parameters
    improvement: ????
    """
    shape = np.shape(images)
    improvement = []  
    
    # Initialise the solution
    coregistered =  np.reshape(images,(shape[0]*shape[1],shape[2]))
    deformation_field = np.zeros((shape[0]*shape[1],2,shape[2]))

    converged = False
    while not converged: 

        # Update the solution
        fit, par = fit_signal_model(shape, coregistered, signal_model_parameters)
        coregistered, new_deformation_field = fit_coregistration(shape, fit, images, image_parameters, elastix_model_parameters)

        ## check convergence    
        improvement.append(maximum_deformation_per_pixel(deformation_field, new_deformation_field))
        converged = improvement[-1] <= precision           
        deformation_field = new_deformation_field

    coregistered = np.reshape(coregistered,(shape[0],shape[1],shape[2]))
    deformation_field = np.reshape(deformation_field,(shape[0],shape[1],2,shape[2])) 
    improvement = pd.DataFrame({'maximum_deformation_per_pixel': improvement}) 

    return coregistered, fit, deformation_field, par, improvement


def fit_signal_model(shape, coregistered, signal_model_parameters):
    """Fit signal model pixel by pixel"""

    fit = np.zeros((shape[0]*shape[1],shape[2]))
    par = np.array([]) 
    for x in range(shape[0]*shape[1]):#pixels
      fit[x,:], par_x = signal_model_fit(coregistered[x,:], signal_model_parameters) 
      par = np.append(par, par_x)
    fit = np.reshape(fit,(shape[0],shape[1],shape[2]))
    return fit, par


def fit_coregistration(shape, fit, images, image_parameters, elastix_model_parameters):
    """Coregister image by image"""

    # Kanishka: Can Elastix MDR be initialised with the solution from the previous iteration?

    coregistered = np.zeros((shape[0]*shape[1],shape[2]))
    deformation_field = np.zeros([shape[0]*shape[1], 2, shape[2]])
    for t in range(shape[2]): #dynamics
      coregistered[:,t], deformation_field[:,:,t] = simpleElastix_MDR_coregistration(images[:,:,t], fit[:,:,t], elastix_model_parameters, image_parameters)
    return coregistered, deformation_field


def maximum_deformation_per_pixel(deformation_field, new_deformation_field):
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
    
    return maximum_deformation_per_pixel, converged


# signal model function for MDR
def signal_model_fit(time_curve, signal_model_parameters): 
    """
        This function takes signal time curve and signal model paramters as input 
        and returns the fitted signal and model parameters
    """
    fit, fitted_parameters = getattr(signal_model_parameters[0][0], signal_model_parameters[0][1])(time_curve, signal_model_parameters) 
    return fit, fitted_parameters


# deformable registration for MDR
def simpleElastix_MDR_coregistration(target, source, elastix_model_parameters, image_parameters):
    """
        This function takes source image and target image as input 
        and returns ffd based co-registered image and deformation field 
    """
    shape_source = np.shape(source)
    shape_target = np.shape(target)

    ## TODO for 3D; OK for 2D images
    source = sitk.GetImageFromArray(source)
    source.SetOrigin(image_parameters[0])
    source.SetSpacing(image_parameters[1])
    source.__SetPixelAsUInt16__
    source = np.reshape(source, [shape_source[0], shape_source[1]]) 
    
    target = sitk.GetImageFromArray(target)
    target.SetOrigin(image_parameters[0])
    target.SetSpacing(image_parameters[1])
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



   