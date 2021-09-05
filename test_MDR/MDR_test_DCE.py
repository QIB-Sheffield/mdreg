"""
MODEL DRIVEN REGISTRATION for iBEAt study: quantitative renal MRI
@Kanishka Sharma 2021

"""
import sys
import numpy as np
import SimpleITK as sitk
from PIL import Image
from pyMDR.MDR import model_driven_registration  
from models  import iBEAt_DCE

np.set_printoptions(threshold=sys.maxsize)

def iBEAt_test_DCE(Elastix_Parameter_file_PATH, output_dir, slice_sorted_acq_time, original_images, slice_parameters, AIFs_PATH, patient_folder):

    ## read sequence acquisition parameter for signal modelling
    aif, times = iBEAt_DCE.load_txt(AIFs_PATH + '/' + str(patient_folder) + '/' + 'AIF__2C Filtration__Curve.txt')
    aif.append(aif[-1])
    times.append(times[-1])
    # select model
    MODEL = [iBEAt_DCE,'fitting']
    # select signal model paramters
    signal_model_parameters = [MODEL, aif, times]
    ## read elastix parameters
    elastixImageFilter = sitk.ElastixImageFilter()
    elastix_model_parameters = elastixImageFilter.ReadParameterFile(Elastix_Parameter_file_PATH + "/BSplines_DCE.txt")
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
        im_motion_corrected.save(output_dir + '/coregistered/MDR-registered_DCE_'+ str(i) + ".tiff")
        im_fit.save(output_dir + '/fit/fit_image_'+ str(i) + ".tiff")
        im_deform_x.save(output_dir + '/deformation_field/final_deformation_x_'+ str(i) + ".tiff")
        im_deform_y.save(output_dir + '/deformation_field/final_deformation_y_'+ str(i) + ".tiff")
        
    ## Fitted Parameters and diagnostics to output folder
    Fp = np.reshape(fitted_quantitative_Params[::4],[shape[0],shape[1]]) 
    Fp_Img = Image.fromarray(Fp)
    Fp_Img.save(output_dir + '/fitted_parameters/Fp' + ".tiff")
    
    Tp = np.reshape(fitted_quantitative_Params[1::4],[shape[0],shape[1]]) 
    Tp_Img = Image.fromarray(Tp)
    Tp_Img.save(output_dir + '/fitted_parameters/Tp' + ".tiff")

    Ps = np.reshape(fitted_quantitative_Params[2::4],[shape[0],shape[1]]) 
    Ps_Img = Image.fromarray(Ps)
    Ps_Img.save(output_dir + '/fitted_parameters/Ps' + ".tiff")

    Te = np.reshape(fitted_quantitative_Params[3::4],[shape[0],shape[1]]) 
    Te_Img = Image.fromarray(Te)
    Te_Img.save(output_dir + '/fitted_parameters/Te' + ".tiff")
    
    diagnostics.to_csv(output_dir + 'DCE_largest_deformations.csv')

    print("Finished processing Model Driven Registration case for iBEAt study DCE sequence!")

        
    
