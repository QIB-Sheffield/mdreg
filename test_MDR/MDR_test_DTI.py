"""
MODEL DRIVEN REGISTRATION for iBEAt study: quantitative renal MRI
@Kanishka Sharma 2021

"""
import sys
import numpy as np
import SimpleITK as sitk
from PIL import Image
from pyMDR.MDR import model_driven_registration  
from models  import iBEAt_DTI

np.set_printoptions(threshold=sys.maxsize)


def iBEAt_test_DTI(Elastix_Parameter_filepath, output_dir, slice_sorted_acq_time, images, image_parameters, fname, lstFilesDCM):
    """ Example application of MDR in renal DTI (iBEAt data)
    
    Parameters
    ----------
    Elastix_Parameter_filepath (string): complete path to the Elastix parameter file to be used
    output_dir (string): directory where results are saved
    slice_sorted_acq_time: WHAT IS THIS KANISHKA??
    images (np array): pixel data formatted as (rows, columns, acquisitions)
    image_parameters (array??? KANISHKA): what is this?
    fname: what is this??
    lstFilesDCM: ????

    Description
    -----------
    What does this function do??
    """

    # Format input variables
    images = sort_images(images, slice_sorted_acq_time)
    signal_model_parameters = signal_model_parameters(fname, lstFilesDCM)
    elastix_model_parameters = elastix_model_parameters(Elastix_Parameter_filepath)
    
    #Perform MDR
    MDR_output = model_driven_registration(images, image_parameters, signal_model_parameters, elastix_model_parameters, precision = 1)
    
    #Export results
    export_images(MDR_output[0], output_dir +'/coregistered/MDR-registered_DTI_')
    export_images(MDR_output[1], output_dir +'/fit/fit_image_')
    export_images(MDR_output[2][:,:,0,:], output_dir +'/deformation_field/final_deformation_x_')
    export_images(MDR_output[2][:,:,1,:], output_dir +'/deformation_field/final_deformation_y_')
    export_maps(MDR_output[3][2::4], output_dir + '/fitted_parameters/FA', np.shape(images))
    export_maps(MDR_output[3][3::4], output_dir + '/fitted_parameters/ADC', np.shape(images))
    MDR_output[4].to_csv(output_dir + 'DTI_largest_deformations.csv')

    print("Finished processing Model Driven Registration case for iBEAt study DTI sequence!")



## read sequence acquisition parameter for signal modelling
def signal_model_parameters(fname, lstFilesDCM):
    b_values, bVec_original, image_orientation_patient = iBEAt_DTI.read_dicom_tags_DTI(fname, lstFilesDCM)
    MODEL = [iBEAt_DTI,'fitting']
    signal_model_parameters = [MODEL, [b_values, bVec_original]]
    signal_model_parameters.append(image_orientation_patient)
    return signal_model_parameters

## read elastix parameters
def elastix_model_parameters(Elastix_Parameter_file_PATH):
    elastixImageFilter = sitk.ElastixImageFilter()
    elastix_model_parameters = elastixImageFilter.ReadParameterFile(Elastix_Parameter_file_PATH + "/BSplines_DTI.txt")
    elastix_model_parameters['MaximumNumberOfIterations'] = ['256'] 
    elastixImageFilter.SetParameterMap(elastix_model_parameters)
    return elastix_model_parameters

## sort original images according to acquisiton times and run MDR
def sort_images(original_images, slice_sorted_acq_time):   
    for i, s in enumerate(slice_sorted_acq_time):
        img2d = s.pixel_array
        original_images[:, :, i] = img2d  
    return original_images 

## Save MDR results to folder
def export_images(MDR_output, folder):
    shape = np.shape(MDR_output)
    for i in range(shape[2]):
        im = Image.fromarray(MDR_output[:,:,i])
        im.save(folder + str(i) + ".tiff")

## Fitted Parameters to output folder
def export_maps(MDR_output, folder, shape):
    array = np.reshape(MDR_output, [shape[0],shape[1]]) 
    Img = Image.fromarray(array)
    Img.save(folder + ".tiff")
    

           


    
