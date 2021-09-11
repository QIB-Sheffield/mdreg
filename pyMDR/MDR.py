"""
MODEL DRIVEN REGISTRATION (MDR) for quantitative renal MRI
MDR Library
@Kanishka Sharma 
@Steven Sourbron
2021
"""

import numpy as np
import SimpleITK as sitk
import pandas as pd

def model_driven_registration(images, image_parameters, signal_model_parameters, elastix_model_parameters, precision = 1): 
    """ Performs model driven registration

    Parameters
    ---------- 
    images: unregistered 2D images (uint16) with shape: [x-dim,y-dim, number of slices]
    image_parameters: SITK input: [image origin, image spacing]
    sitk Origin: location in the world coordinate system of the voxel 
    example origin: (-186.70486942826, 33.200800281755, 241.22697840639); 
    sitk Spacing: distance between pixels along each of the dimensions
    example spacing (mm): (1.0416666269302, 1.0416666269302, 1.0)
    signal_model_parameters: [MODEL, model specific parameters]
    elastix_model_parameters: elastix file registration parameters
    precision: in mm 

    Return values
    ------------
    coregistered: ffd based co-registered (uint16) 2D images with shape: [x-dim,y-dim, number of slices]
    fit: signal model fit image
    fit image dimension = 2D with shape: [x-dim,y-dim, num of slices]
    fit image type: uint16
    output deformation fields: deformation_field_x, deformation_field_y
    deformation field dimension: [x-dim, y-dim, 2, num of slices]
    par: fitted parameters
    improvement: maximum deformation per pixel calculated as the euclidean distance of difference between old and new deformation field
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

    coregistered = np.zeros((shape[0]*shape[1],shape[2]))
    deformation_field = np.zeros([shape[0]*shape[1], 2, shape[2]])
    for t in range(shape[2]): #dynamics
      coregistered[:,t], deformation_field[:,:,t] = simpleElastix_MDR_coregistration(images[:,:,t], fit[:,:,t], elastix_model_parameters, image_parameters)
    return coregistered, deformation_field


def maximum_deformation_per_pixel(deformation_field, new_deformation_field):
    """
    This function calculates diagnostics from the registration process
    It takes as input the original deformation field and the new deformation field
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
def simpleElastix_MDR_coregistration(target, source, elastix_model_parameters, image_parameters):
    """
        This function takes unregistered source image and target image as input 
        and returns ffd based co-registered image and corresponding deformation field 
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



   