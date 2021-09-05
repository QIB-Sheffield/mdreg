"""
MODEL DRIVEN REGISTRATION for iBEAt study: quantitative renal MRI
@Kanishka Sharma 2021

"""
import sys
import numpy as np
import SimpleITK as sitk
from PIL import Image
from pyMDR.MDR import model_driven_registration  
from models  import iBEAt_T2star

np.set_printoptions(threshold=sys.maxsize)

def iBEAt_test_T2star(Elastix_Parameter_file_PATH, output_dir, slice_sorted_acq_time, original_images, slice_parameters, fname, lstFilesDCM):

    ## ## read sequence acquisition parameter for signal modelling
    echo_times, slice_sorted_echo_time = iBEAt_T2star.read_echo_times(fname, lstFilesDCM)
    # select model
    MODEL = [iBEAt_T2star,'fitting']
    # select signal model paramters
    signal_model_parameters = [MODEL, echo_times]
    ## read elastix parameters
    elastixImageFilter = sitk.ElastixImageFilter()
    elastix_model_parameters = elastixImageFilter.ReadParameterFile(Elastix_Parameter_file_PATH + "/BSplines_T2star.txt")
    elastix_model_parameters['MaximumNumberOfIterations'] = ['256'] 
    elastixImageFilter.SetParameterMap(elastix_model_parameters) 
    
    ## sort original images according to acquisiton times and run MDR
    for i, s in enumerate(slice_sorted_acq_time):
        img2d = s.pixel_array
        original_images[:, :, i] = img2d
        
    shape = np.shape(original_images)

    MDR_output = []
                    
    MDR_output = model_driven_registration(original_images, slice_parameters, signal_model_parameters, elastix_model_parameters, precision  = 1)
    
    # MDR output variables 
    motion_corrected_images = MDR_output[0]
    fit = MDR_output[1]
    deformation_field = MDR_output[2]
    fitted_quantitative_Params = MDR_output[3]
    diagnostics = MDR_output[4]
    
    ## create new arrays to store final results to output folder   
    im_motion_corrected = np.zeros([shape[0],shape[1]], dtype=np.uint16) #
    im_fit = np.zeros([shape[0],shape[1]], dtype=np.uint16) #
    im_deform_x = np.zeros([shape[0],shape[1],2], dtype=np.uint16)
    im_deform_y = np.zeros([shape[0],shape[1],2], dtype=np.uint16)

    ## Save MDR results to folder
    for i in range(shape[2]):
        im_motion_corrected = Image.fromarray(motion_corrected_images[:,:,i])
        im_fit = Image.fromarray(fit[:,:,i])
        im_deform_x = Image.fromarray(deformation_field[:,:,0,i])
        im_deform_y = Image.fromarray(deformation_field[:,:,1,i])
        im_motion_corrected.save(output_dir + '/coregistered/MDR-registered_T2star_dynamic_'+ str(i) + ".tiff")
        im_fit.save(output_dir + '/fit/fit_image_'+ str(i) + ".tiff")
        im_deform_x.save(output_dir + '/deformation_field/final_deformation_x_'+ str(i) + ".tiff")
        im_deform_y.save(output_dir + '/deformation_field/final_deformation_y_'+ str(i) + ".tiff")
        
    ## Fitted Parameters and diagnostics to output folder
    S0_T2starMap = np.reshape(fitted_quantitative_Params[::2],[shape[0],shape[1]]) 
    S0_T2starMap_Img = Image.fromarray(S0_T2starMap)
    S0_T2starMap_Img.save(output_dir + '/fitted_parameters/S0_T2star_Map' + ".tiff")
    
    T2starMap = np.reshape(fitted_quantitative_Params[1::2],[shape[0],shape[1]]) 
    T2starMap_Img = Image.fromarray(T2starMap)
    T2starMap_Img.save(output_dir + '/fitted_parameters/T2star_Map' + ".tiff")
    
    diagnostics.to_csv(output_dir + 'T2star_largest_deformations.csv')

    print("Finished processing Model Driven Registration case for iBEAt study T2* mapping sequence!")

 


    
