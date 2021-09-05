"""
MODEL DRIVEN REGISTRATION for iBEAt study: quantitative renal MRI
@Kanishka Sharma 2021

"""
import sys
import glob
import os
import numpy as np
import SimpleITK as sitk
from PIL import Image
import pydicom
from collections import OrderedDict
from pathlib import Path 
from pyMDR.MDR import model_driven_registration  
from models  import   iBEAt_T1, iBEAt_T2, iBEAt_T2star, iBEAt_DTI, iBEAt_DWI_monoexponential, iBEAt_DCE

np.set_printoptions(threshold=sys.maxsize)


if __name__ == '__main__':
  
    ## select sequence to process
    SEQUENCES = ['T2']
    CORRESPONDANCE = OrderedDict()
    CORRESPONDANCE['T1'] = ['T1', 140, 5] 
    CORRESPONDANCE['T2'] = ['T2', 55, 5]
    CORRESPONDANCE['T2star'] = ['T2star', 60, 5]
    CORRESPONDANCE['DTI'] = ['DTI', 4380, 30]    
    CORRESPONDANCE['IVIM'] = ['IVIM', 900, 30]  
    CORRESPONDANCE['DCE'] = ['DCE', 2385, 9] 

    ## path definitions   
    ## your 'os.getcwd()' path should point to your local MDR_Library directory
    print(os.getcwd()) # eg: /Users/kanishkasharma/Documents/GitHub/MDR_Library
    DATA_PATH = os.getcwd() + r'/test_MDR/test_data/DICOMs'
    AIFs_PATH = os.getcwd() + r'/test_MDR/test_data/AIFs' 
    OUTPUT_REG_PATH = os.getcwd() + r'/test_MDR/iBEAt_MDR/MDR_registration_output'
    Elastix_Parameter_file_PATH = os.getcwd() + r'/Elastix_Parameters_Files/iBEAt'

    print(" user test case for Model Driven Registration")
    lstFilesDCM = []  

    # Organize files per each sequence:
    for sequence in SEQUENCES:
        folder = CORRESPONDANCE[sequence][0]
        num_of_files = CORRESPONDANCE[sequence][1]
        slices = CORRESPONDANCE[sequence][2]
       
        os.chdir(DATA_PATH)    
        patients_folders = os.listdir()
        print(patients_folders) 
        for patient_folder in patients_folders:
               if patient_folder not in ['test_case_iBEAt_4128009']: 
                  continue
               print("Processing iBEAt MDR test case:",patient_folder) 
               sequence_images_path = patient_folder + '/' + str(folder) + '/DICOM'
               print(sequence_images_path)
               os.chdir(DATA_PATH + '/' + sequence_images_path)
            
               dcm_files_found = glob.glob("*.dcm")
               if not dcm_files_found:
                  dcm_files_found = glob.glob("*.IMA")
            
               for slice in range(1, slices+1):
                    current_slice = sequence + '_slice_' + str(slice)
                    ## single slice processing per sequence for test
                    if sequence == 'T1' and current_slice not in [sequence + '_slice_4']: 
                        continue
                    elif sequence == 'T2' and current_slice not in [sequence + '_slice_3']:
                        continue
                    elif sequence == 'T2star' and current_slice not in [sequence + '_slice_3']:
                        continue
                    elif sequence == 'DTI' and current_slice not in [sequence + '_slice_15']:
                        continue
                    elif sequence == 'IVIM' and current_slice not in [sequence + '_slice_15']:
                        continue
                    elif sequence == 'MT' and current_slice not in [sequence + '_slice_8']: 
                        continue
                    elif sequence == 'DCE' and current_slice not in [sequence + '_slice_5']:
                        continue
                    
                    slice_path = DATA_PATH + '/' + sequence_images_path + '/' + current_slice

                    data = Path(slice_path)

                    lstFilesDCM = list(data.glob('**/*.IMA')) 
            
                    files = []

                    slice_parameters = []

                    for fname in lstFilesDCM:
                        print("loading: {}".format(fname))
                        files.append(pydicom.dcmread(fname))

                    RefDs = pydicom.dcmread(lstFilesDCM[0])

                    SeriesPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))           
                    ArrayDicomiBEAt = np.zeros(SeriesPixelDims, dtype=RefDs.pixel_array.dtype)
  
                    print("file count: {}".format(len(files)))

                    # loop through all the DICOM slices
                    for filenameDCM in lstFilesDCM:
                        # read the file
                        ds = pydicom.dcmread(filenameDCM)
                        # write pixel data into numpy array
                        ArrayDicomiBEAt[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array  
                    
                    ## Get Pixel details from DICOM
                    reader = sitk.ImageSeriesReader()
                    dicom_names = reader.GetGDCMSeriesFileNames(slice_path)
                    reader.SetFileNames(dicom_names)
                    image = reader.Execute()
                    image_array = sitk.GetArrayFromImage(image) 
                    origin = image.GetOrigin() 
                    spacing = image.GetSpacing() 

                    # Sort all images according to the acquisition time
                    slice_sorted_acq_time = []
                   
                    skipcount = 0
                    for f in files: 
                       if hasattr(f, 'AcquisitionTime'):
                            slice_sorted_acq_time.append(f)
                            
                       else:
                            skipcount = skipcount + 1

                    print("skipped, no AcquisitionTime: {}".format(skipcount))

                    slice_sorted_acq_time = sorted(slice_sorted_acq_time, key=lambda s: s.AcquisitionTime)
                     
                    slice_parameters = [origin, spacing]

                    img_shape = np.shape(ArrayDicomiBEAt)

                    original_images = np.zeros(img_shape)
  
                    ## MDR for T2-mapping sequence
                    if sequence == 'T2':

                        ## output directory for final registration results 
                        output_dir =  OUTPUT_REG_PATH + '/T2/'
                        ## read sequence acquisition parameter for signal modelling
                        T2_prep_times = iBEAt_T2.read_prep_times()
                        # select model
                        MODEL = [iBEAt_T2,'fitting'] 
                        # select signal model paramters
                        signal_model_parameters = [MODEL, T2_prep_times]
                        ## read elastix parameters
                        elastixImageFilter = sitk.ElastixImageFilter()
                        # TODO: read parameters file before this
                        elastix_model_parameters = elastixImageFilter.ReadParameterFile(Elastix_Parameter_file_PATH + "/BSplines_T2.txt")
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
                            im_motion_corrected.save(output_dir + '/coregistered/MDR-registered_T2_dynamic_'+ str(i) + ".tiff")
                            im_fit.save(output_dir + '/fit/fit_image_'+ str(i) + ".tiff")
                            im_deform_x.save(output_dir + '/deformation_field/final_deformation_x_'+ str(i) + ".tiff")
                            im_deform_y.save(output_dir + '/deformation_field/final_deformation_y_'+ str(i) + ".tiff")
                            
                        ## Fitted Parameters and daignostics to output folder
                        S0_T2 = np.reshape(fitted_quantitative_Params[::2],[shape[0],shape[1]]) 
                        S0_T2_Img = Image.fromarray(S0_T2)
                        S0_T2_Img.save(output_dir + '/fitted_parameters/S0_T2_Map' + ".tiff")
                        
                        T2_Map = np.reshape(fitted_quantitative_Params[1::2],[shape[0],shape[1]]) 
                        T2_Map_Img = Image.fromarray(T2_Map)
                        T2_Map_Img.save(output_dir + '/fitted_parameters/T2_Map' + ".tiff")
                        
                        diagnostics.to_csv(output_dir + 'T2_largest_deformations.csv')

                        print("Finished processing Model Driven Registration case for iBEAt study T2 mapping sequence!")

                    else:
                        raise Exception("iBEAt sequence not recognised")





    
