"""
MODEL DRIVEN REGISTRATION for iBEAt study: quantitative renal MRI
@Kanishka Sharma 2021

"""
import sys
sys.path.insert(1, '/Users/kanishkasharma/Documents/GitHub')
import glob
import os
import shutil

import numpy as np
import SimpleITK as sitk
from PIL import Image
import pydicom
from collections import OrderedDict
from pathlib import Path 
from MDR_Library.pyMDR.MDR import model_driven_registration  
from MDR_Library.models  import   iBEAt_T1, iBEAt_T2, iBEAt_T2star, iBEAt_DTI, iBEAt_IVIM_monoexponential, iBEAt_DCE
from matplotlib import pyplot as plt

np.set_printoptions(threshold=sys.maxsize)

############################################


if __name__ == '__main__':
  
    CORRESPONDANCE = OrderedDict()
    CORRESPONDANCE['T1'] = ['T1', 140, 5] 
    CORRESPONDANCE['T2'] = ['T2', 55, 5]
    CORRESPONDANCE['T2star'] = ['T2star', 60, 5]
    CORRESPONDANCE['DTI'] = ['DTI', 4380, 30]    
    CORRESPONDANCE['IVIM'] = ['IVIM', 900, 30]  
    CORRESPONDANCE['DCE'] = ['DCE', 2385, 9] # does not work if folder number changes eg: 39, 42, etc.

    print(os.getcwd())
    DATA_PATH = os.getcwd() + r'/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/data/DICOMs'
    AIFs_PATH = os.getcwd() + r'/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/data/AIFs' 
    OUTPUT_REG_PATH = os.getcwd() + r'/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final'

    print("iBEAt Model Driven Registration")
    lstFilesDCM = []  

    # Reorganize files per sequence:
    for sequence in SEQUENCES:
        folder = CORRESPONDANCE[sequence][0]
        num_of_files = CORRESPONDANCE[sequence][1]
        slices = CORRESPONDANCE[sequence][2]
       
        os.chdir(DATA_PATH)    
        patients_folders = os.listdir()
        print(patients_folders) # ['.DS_Store', 'Leeds_Patient_4128009']
        for patient_folder in patients_folders:
               if patient_folder not in ['Leeds_Patient_4128009']:
                  continue
           # if patient_folder!= '.DS_Store':
               print("Processing iBEAt case:",patient_folder) # Processing iBEAt case: .DS_Store
               sequence_images_path = patient_folder + '/' + str(folder) + '/DICOM'
               print(sequence_images_path)
               os.chdir(DATA_PATH + '/' + sequence_images_path)
            
               # Make sure that no acquisition is missing:
               dcm_files_found = glob.glob("*.dcm")
               if not dcm_files_found:
                  dcm_files_found = glob.glob("*.IMA")
            
            
               for slice in range(1, slices+1):
                    current_slice = sequence + '_slice_' + str(slice)
                    ##TODO: Delete this to work for all slices and sequences
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
                    elif sequence == 'MT' and current_slice not in [sequence + '_slice_8']: # TBC
                        continue
                    elif sequence == 'DCE' and current_slice not in [sequence + '_slice_5']:
                        continue
                    
                    slice_path = DATA_PATH + '/' + sequence_images_path + '/' + current_slice

                    data = Path(slice_path)

                    lstFilesDCM = list(data.glob('**/*.IMA')) # NOTE: these are not sorted correctly 
            
                    files = []

                    slice_parameters = []
                
                    for fname in lstFilesDCM:
                        print("loading: {}".format(fname))
                        files.append(pydicom.dcmread(fname))

                    RefDs = pydicom.dcmread(lstFilesDCM[0])
                    # Load spacing (in mm)
                    SeriesPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness)) 
                    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
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
                    print("origin")
                    print(origin)
                    spacing = image.GetSpacing() 
                    print("spacing")
                    print(spacing)
                    slice_direction = image.GetDirection() 
                    print("slice_direction")
                    print(slice_direction)
                    pixel_type = image.GetPixelIDTypeAsString()
                    print("pixel_type")
                    print(pixel_type)
                    
                    slice_sorted_acq_time = []
                    # skip files with no acq time
                   
                    skipcount = 0
                    for f in files: 
                       if hasattr(f, 'AcquisitionTime'):
                            slice_sorted_acq_time.append(f)
                            
                       else:
                            skipcount = skipcount + 1

                    print("skipped, no AcquisitionTime: {}".format(skipcount))

                    slice_sorted_acq_time = sorted(slice_sorted_acq_time, key=lambda s: s.AcquisitionTime)
                   
                    # ## extract pixel-spacing from dicom header
                    ps = slice_sorted_acq_time[0].PixelSpacing
 
                    # ##extract slice thickness from dicom header
                    slice_thickness = slice_sorted_acq_time[0].SliceThickness
                    #print(slice_thickness)
                    
                    slice_parameters = [origin, spacing, pixel_type, slice_thickness, slice_direction]

                    img_shape = np.shape(ArrayDicomiBEAt)#
                    original_images = np.zeros(img_shape)
                    selected_image = np.zeros(img_shape)


                    if sequence == 'T1':

                        ## read sequence related parameters
                        inversion_times, slice_sorted_inv_time = iBEAt_T1.read_inversion_times_and_sort(fname, lstFilesDCM)

                        for i, s in enumerate(slice_sorted_inv_time):
                            img2d = s.pixel_array
                            original_images[:, :, i] = img2d
                            resaved = Image.fromarray(img2d)
                            resaved.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/T1/original_data_resaved/original_resaved_'+ str(i) + ".tiff")
 
                        MODEL = [iBEAt_T1,'fitting'] 
                        
                        signal_model_parameters = [MODEL, inversion_times]

                        elastixImageFilter = sitk.ElastixImageFilter()
                        elastix_model_parameters = elastixImageFilter.ReadParameterFile("/Users/kanishkasharma/Documents/GitHub/MDR_Library/Elastix_Parameters_Files/iBEAt/BSplines_T1.txt")
                        elastix_model_parameters['MaximumNumberOfIterations'] = ['256']
                        elastixImageFilter.SetParameterMap(elastix_model_parameters) 
                        elastixImageFilter.PrintParameterMap()
                      
                        output_dir =  OUTPUT_REG_PATH + '/T1/'

                        ## full image analysis
                        shape = np.shape(original_images)
                        
                        original_images_reslice = np.zeros([shape[0]*shape[1], shape[2]], dtype=np.uint16) #

                        for i in range(shape[2]):#dynamics
                            reslice = sitk.GetImageFromArray(original_images[:,:,i])
                            reslice.SetOrigin(slice_parameters[0])
                            reslice.SetSpacing(slice_parameters[1])
                            reslice.__SetPixelAsUInt16__
                            original_images_reslice[:,i]  = reslice

                        original_images_reslice = np.reshape(original_images_reslice, [shape[0], shape[1], shape[2]]) # 

                        im_original = np.zeros([shape[0],shape[1]], dtype=np.uint16) 

                        for i in range(shape[2]):
                            im_original = Image.fromarray(original_images_reslice[:,:,i])
                            im_original.save('//Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/T1/original/original_series_'+ str(i) + ".tiff")

                        ## for full image analysis                 
                        motion_corrected_images, fit, deformation_field, fitted_quantitative_Params, diagnostics = model_driven_registration(original_images, slice_parameters, signal_model_parameters, elastix_model_parameters, precision  = 1)
                        
                        diagnostics.to_csv('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/T1/T1_largest_deformations.csv' , index=False)
                          
                        im_motion_corrected = np.zeros([shape[0],shape[1]], dtype=np.uint16) #
                        im_fit = np.zeros([shape[0],shape[1]], dtype=np.uint16) #
                        im_deform_x = np.zeros([shape[0],shape[1],2], dtype=np.uint16)
                        im_deform_y = np.zeros([shape[0],shape[1],2], dtype=np.uint16)

                        for i in range(shape[2]):
                            im_motion_corrected = Image.fromarray(motion_corrected_images[:,:,i])
                            im_fit = Image.fromarray(fit[:,:,i])
                            im_deform_x = Image.fromarray(deformation_field[:,:,0,i])
                            im_deform_y = Image.fromarray(deformation_field[:,:,1,i])
                            im_motion_corrected.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/T1/coregistered/co-registered_'+ str(i) + ".tiff")
                            im_fit.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/T1/fit/final_fit_'+ str(i) + ".tiff")
                            im_deform_x.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/T1/final_deformation_x_/final_deformation_x_' + str(i) + ".tiff")
                            im_deform_y.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/T1/final_deformation_y_/final_deformation_y_' + str(i) + ".tiff")
                        
                      
                        print("fitted_quantitative params shape")
                        print(np.shape(fitted_quantitative_Params)) ## 1-D list
 
                        T1_estimated = np.reshape(fitted_quantitative_Params[::4],[shape[0],shape[1]]) #
                        T1_estimated_Img = Image.fromarray(T1_estimated)
                        T1_estimated_Img.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/T1/fitted_params/T1_estimated' + ".tiff")
                        
                        T1_apparent = np.reshape(fitted_quantitative_Params[1::4],[shape[0],shape[1]]) 
                        T1_apparent_Img = Image.fromarray(T1_apparent)
                        T1_apparent_Img.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/T1/fitted_params/T1_apparent' + ".tiff")
                       
                        B = np.reshape(fitted_quantitative_Params[2::4],[shape[0],shape[1]])
                        B_Img = Image.fromarray(B)
                        B_Img.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/T1/fitted_params/B' + ".tiff")
                        
                        A = np.reshape(fitted_quantitative_Params[3::4],[shape[0],shape[1]])
                        A_Img = Image.fromarray(A)
                        A_Img.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/T1/fitted_params/A' + ".tiff")
                        
                        plt.imshow(T1_estimated)
                        plt.colorbar()
                        plt.show()  

                        ## visualise results
                        plt.imshow(original_images[:,:,3].squeeze(), cmap="gray")
                        plt.colorbar()
                        plt.show()  
                        
                        plt.imshow(motion_corrected_images[:,:,3].squeeze(), cmap="gray")
                        plt.colorbar()
                        plt.show() 


                    elif sequence == 'T2':

                        ## read sequence related parameters
                        T2_prep_times = iBEAt_T2.read_prep_times()

                        for i, s in enumerate(slice_sorted_acq_time):
                            img2d = s.pixel_array
                            original_images[:, :, i] = img2d
                            resaved = Image.fromarray(img2d)
                            ## TODO change according to series
                            resaved.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/T2/original_data_resaved/original_resaved_'+ str(i) + ".tiff")

                        ## crop image for test (Y,X)
                        image_arr = original_images[70:290, 50:200, :] # # covers right kidney 009 
                        shape = np.shape(image_arr)

                        MODEL = [iBEAt_T2,'fitting'] #filename and function name
                        
                        signal_model_parameters = [MODEL, T2_prep_times]#, [image_orientation_patient]]
                        print("signal_model_parameters")
                        print(signal_model_parameters)

                        ## read elastix params
                        elastixImageFilter = sitk.ElastixImageFilter()
                        elastix_model_parameters = elastixImageFilter.ReadParameterFile("/Users/kanishkasharma/Documents/GitHub/MDR_Library/Elastix_Parameters_Files/iBEAt/BSplines_T2.txt")
                        elastix_model_parameters['MaximumNumberOfIterations'] = ['256'] 
                        elastixImageFilter.SetParameterMap(elastix_model_parameters) 
                        elastixImageFilter.PrintParameterMap()
                        
                        output_dir =  OUTPUT_REG_PATH + '/T2/' ##ok

                        ## full image analysis
                        shape = np.shape(original_images)

                        original_images_reslice = np.zeros([shape[0]*shape[1], shape[2]], dtype=np.uint16) #

                        for i in range(shape[2]):#dynamics
                            reslice = sitk.GetImageFromArray(original_images[:,:,i])
                            reslice.SetOrigin(slice_parameters[0])
                            reslice.SetSpacing(slice_parameters[1])
                            reslice.__SetPixelAsUInt16__
                            original_images_reslice[:,i]  = reslice

                        original_images_reslice = np.reshape(original_images_reslice, [shape[0], shape[1], shape[2]]) 
                        print("original_images image shape")
                        print(np.shape(original_images_reslice))

                        im_original = np.zeros([shape[0],shape[1]], dtype=np.uint16) 

                        for i in range(shape[2]):
                            im_original = Image.fromarray(original_images_reslice[:,:,i])
                            im_original.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/T2/original/original_series_'+ str(i) + ".tiff")

                        ## uncomment below for full image analysis                 
                        #motion_corrected_images, fit, deformation_field, fitted_quantitative_Params, diagnostics = model_driven_registration(original_images, slice_parameters, signal_model_parameters, elastix_model_parameters, precision  = 1)
                        
                        # plt.imshow(original_images[:,:,47].squeeze(), cmap="gray")
                        # plt.colorbar()
                        # plt.show()

                        ## cropped image analysis; ## 1mm precision as pixel size is 1.04mm
                        ## TODO: comment line below for full image analysis
                        
                        cropped_images_reslice = np.zeros([shape_image_arr[0]*shape_image_arr[1], shape_image_arr[2]], dtype=np.uint16) #

                        for i in range(shape_image_arr[2]):#dynamics
                            reslice = sitk.GetImageFromArray(image_arr[:,:,i])
                            reslice.SetOrigin(slice_parameters[0])
                            reslice.SetSpacing(slice_parameters[1])
                            reslice.__SetPixelAsUInt16__
                            cropped_images_reslice[:,i]  = reslice

                        cropped_images_reslice = np.reshape(cropped_images_reslice, [shape_image_arr[0], shape_image_arr[1], shape_image_arr[2]]) # required as the above GetImageFromArray reshapes the source to (147456,)
                        print("cropped image shape")
                        print(np.shape(cropped_images_reslice))

                        im_crop = np.zeros([shape_image_arr[0],shape_image_arr[1]], dtype=np.uint16) 

                        for i in range(shape_image_arr[2]):
                            im_crop = Image.fromarray(cropped_images_reslice[:,:,i])
                            im_crop.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/T2/original_data_cropped/original_cropped_resliced_'+ str(i) + ".tiff")
        
                        #sys.exit()
                        motion_corrected_images, fit, deformation_field, fitted_quantitative_Params, diagnostics = model_driven_registration(cropped_images_reslice, slice_parameters, signal_model_parameters, elastix_model_parameters, precision  = 1) 

                        diagnostics.to_csv('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/T2/T2_largest_deformations.csv' , index=False)
                            
                        im_motion_corrected = np.zeros([shape[0],shape[1]], dtype=np.uint16) #
                        im_fit = np.zeros([shape[0],shape[1]], dtype=np.uint16) #
                        im_deform_x = np.zeros([shape[0],shape[1],2], dtype=np.uint16)
                        im_deform_y = np.zeros([shape[0],shape[1],2], dtype=np.uint16)

                        for i in range(shape[2]):
                            im_motion_corrected = Image.fromarray(motion_corrected_images[:,:,i])
                            im_fit = Image.fromarray(fit[:,:,i])
                            im_deform_x = Image.fromarray(deformation_field[:,:,0,i])
                            im_deform_y = Image.fromarray(deformation_field[:,:,1,i])
                            im_motion_corrected.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/T2/coregistered/co-registered_test_'+ str(i) + ".tiff")
                            im_fit.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/T2/fit/final_fit_'+ str(i) + ".tiff")
                            im_deform_x.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/T2/final_deformation_x_/final_deformation_x_' + str(i) + ".tiff")
                            im_deform_y.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/T2/final_deformation_y_/final_deformation_y_' + str(i) + ".tiff")
                        
                        print("fitted_quantitative params shape")
                        print(np.shape(fitted_quantitative_Params)) ## 1-D list
                        
                        ## Fitted Parameters:fitted_parameters = T2starMap
                        S0_T2_Map = np.reshape(fitted_quantitative_Params[::2],[shape[0],shape[1]]) 
                        S0_T2_Map_Img = Image.fromarray(S0_T2_Map)
                        S0_T2_Map_Img.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/T2/fitted_params/S0_T2_Map' + ".tiff")
                    

                        T2_Map = np.reshape(fitted_quantitative_Params[1::2],[shape[0],shape[1]]) 
                        T2_Map_Img = Image.fromarray(T2_Map)
                        T2_Map_Img.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/T2/fitted_params/T2_Map' + ".tiff")
                    
                        c =plt.imshow(T2_Map, cmap="gray", extent =[10, 80, 10, 80])
                        plt.colorbar(c)
                        plt.show()  

                        ## visualise results
                        plt.imshow(original_images[:,:,3].squeeze(), cmap="gray")
                        plt.colorbar()
                        plt.show()  
                        
                        plt.imshow(motion_corrected_images[:,:,3].squeeze(), cmap="gray")
                        plt.colorbar()
                        plt.show() 

  
                    elif sequence == 'T2star':

                        ## read sequence related parameters
                        echo_times, slice_sorted_echo_time = iBEAt_T2star.read_echo_times(fname, lstFilesDCM)
                        print(" echo_times")
                        print(echo_times)

                        for i, s in enumerate(slice_sorted_echo_time):
                            img2d = s.pixel_array
                            original_images[:, :, i] = img2d
                            resaved = Image.fromarray(img2d)
                            resaved.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/T2star/original_data_resaved/original_resaved_'+ str(i) + ".tiff")


                        ## crop image for test (Y,X)
                        image_arr = original_images[120:360, 50:230, :] # # covers right kidney 009 
                        shape = np.shape(image_arr) ## (240, 180, 12)
                        shape_image_arr = np.shape(image_arr)
                        #print(shape_image_arr)

                        MODEL = [iBEAt_T2star,'fitting'] #filename and function name
                        
                        signal_model_parameters = [MODEL, echo_times]#, [image_orientation_patient]]
                        print("signal_model_parameters")
                        print(signal_model_parameters)
                        #sys.exit()
                        ## read elastix params
                        elastixImageFilter = sitk.ElastixImageFilter()
                        elastix_model_parameters = elastixImageFilter.ReadParameterFile("/Users/kanishkasharma/Documents/GitHub/MDR_Library/Elastix_Parameters_Files/iBEAt/BSplines_T2star.txt")
                        elastix_model_parameters['MaximumNumberOfIterations'] = ['256'] ##['2'] # test with 2 only
                        elastixImageFilter.SetParameterMap(elastix_model_parameters) 
                        elastixImageFilter.PrintParameterMap()
                      
                        output_dir =  OUTPUT_REG_PATH + '/T2star/' ##ok

                        ## full image analysis
                        shape = np.shape(original_images)
 
                        original_images_reslice = np.zeros([shape[0]*shape[1], shape[2]], dtype=np.uint16) #

                        for i in range(shape[2]):#dynamics
                            reslice = sitk.GetImageFromArray(original_images[:,:,i])
                            reslice.SetOrigin(slice_parameters[0])
                            reslice.SetSpacing(slice_parameters[1])
                            reslice.__SetPixelAsUInt16__
                            original_images_reslice[:,i]  = reslice

                        original_images_reslice = np.reshape(original_images_reslice, [shape[0], shape[1], shape[2]]) # required as the above GetImageFromArray reshapes the source to (147456,)
                        print("original_images image shape")
                        print(np.shape(original_images_reslice))

                        im_original = np.zeros([shape[0],shape[1]], dtype=np.uint16) 

                        for i in range(shape[2]):
                            im_original = Image.fromarray(original_images_reslice[:,:,i])
                            im_original.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/T2star/original/original_series_'+ str(i) + ".tiff")

                        ## uncomment below for full image analysis                 
                        #motion_corrected_images, fit, deformation_field, fitted_quantitative_Params, diagnostics = model_driven_registration(original_images, slice_parameters, signal_model_parameters, elastix_model_parameters, precision  = 1)
                        
                        # plt.imshow(original_images[:,:,47].squeeze(), cmap="gray")
                        # plt.colorbar()
                        # plt.show()

                        ## cropped image analysis; ## 1mm precision as pixel size is 1.04mm
                        ## TODO: comment line below for full image analysis
                        ## correct the pixel resolution for cropped region
                        
                        cropped_images_reslice = np.zeros([shape_image_arr[0]*shape_image_arr[1], shape_image_arr[2]], dtype=np.uint16) #

                        for i in range(shape_image_arr[2]):#dynamics
                            reslice = sitk.GetImageFromArray(image_arr[:,:,i])
                            reslice.SetOrigin(slice_parameters[0])
                            reslice.SetSpacing(slice_parameters[1])
                            reslice.__SetPixelAsUInt16__
                            cropped_images_reslice[:,i]  = reslice

                        cropped_images_reslice = np.reshape(cropped_images_reslice, [shape_image_arr[0], shape_image_arr[1], shape_image_arr[2]]) # required as the above GetImageFromArray reshapes the source to (147456,)
                        print("cropped image shape")
                        print(np.shape(cropped_images_reslice))

                        im_crop = np.zeros([shape_image_arr[0],shape_image_arr[1]], dtype=np.uint16) 

                        for i in range(shape_image_arr[2]):
                            im_crop = Image.fromarray(cropped_images_reslice[:,:,i])
                            im_crop.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/T2star/original_data_cropped/original_cropped_resliced_'+ str(i) + ".tiff")
        
                        #sys.exit()
                        motion_corrected_images, fit, deformation_field, fitted_quantitative_Params, diagnostics = model_driven_registration(cropped_images_reslice, slice_parameters, signal_model_parameters, elastix_model_parameters, precision  = 1) 

                        diagnostics.to_csv('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/T2star/T2star_largest_deformations.csv' , index=False)
                          
                        im_motion_corrected = np.zeros([shape[0],shape[1]], dtype=np.uint16) #
                        im_fit = np.zeros([shape[0],shape[1]], dtype=np.uint16) #
                        im_deform_x = np.zeros([shape[0],shape[1],2], dtype=np.uint16)
                        im_deform_y = np.zeros([shape[0],shape[1],2], dtype=np.uint16)

                        for i in range(shape[2]):
                            im_motion_corrected = Image.fromarray(motion_corrected_images[:,:,i])
                            im_fit = Image.fromarray(fit[:,:,i])
                            im_deform_x = Image.fromarray(deformation_field[:,:,0,i])
                            im_deform_y = Image.fromarray(deformation_field[:,:,1,i])
                            im_motion_corrected.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/T2star/coregistered/co-registered_test_'+ str(i) + ".tiff")
                            im_fit.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/T2star/fit/final_fit_'+ str(i) + ".tiff")
                            im_deform_x.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/T2star/final_deformation_x_/final_deformation_x_' + str(i) + ".tiff")
                            im_deform_y.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/T2star/final_deformation_y_/final_deformation_y_' + str(i) + ".tiff")
                        
                        print("fitted_quantitative params shape")
                        print(np.shape(fitted_quantitative_Params)) ## 1-D list
                        #fitted_parameters = [T2starMap]

                        shape = np.shape(image_arr) #only for test

                         ## Fitted Parameters:fitted_parameters = T2starMap
                        S0_T2starMap = np.reshape(fitted_quantitative_Params[::2],[shape[0],shape[1]]) #
                        S0_T2starMap_Img = Image.fromarray(S0_T2starMap)
                        S0_T2starMap_Img.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/T2star/fitted_params/S0_T2starMap' + ".tiff")
                    

                        T2starMap = np.reshape(fitted_quantitative_Params[1::2],[shape[0],shape[1]]) #
                        T2starMap_Img = Image.fromarray(T2starMap)
                        T2starMap_Img.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/T2star/fitted_params/T2starMap' + ".tiff")
                    
                        plt.imshow(T2starMap, extent =[10, 100, 10, 100])
                        plt.colorbar()
                        plt.show()  

                        ## visualise results
                        plt.imshow(original_images[:,:,3].squeeze(), cmap="gray")
                        plt.colorbar()
                        plt.show()  
                        
                        plt.imshow(motion_corrected_images[:,:,3].squeeze(), cmap="gray")
                        plt.colorbar()
                        plt.show() 


                    elif sequence == 'DTI':

                        for i, s in enumerate(slice_sorted_acq_time):
                            img2d = s.pixel_array
                            original_images[:, :, i] = img2d
                            resaved = Image.fromarray(img2d)
                            ## TODO change according to series
                            resaved.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/DTI/original_data_resaved/original_resaved_'+ str(i) + ".tiff")


                         ## crop image for test (Y,X)
                        image_arr = original_images[20:130, 20:160, :] #
                      
                        shape_image_arr = np.shape(image_arr)

                        ## read sequence related parameters
                        b_values, bVec_original, image_orientation_patient = iBEAt_DTI.read_dicom_tags_DTI(fname, lstFilesDCM)
                        print("b_values")
                        print(b_values)
                        print("bVec_original")
                        print(bVec_original)
                        print("image_orientation_patient")
                        print(image_orientation_patient) 
                       # sys.exit()

                        MODEL = [iBEAt_DTI,'fitting'] #filename and function name
                        
                        signal_model_parameters = [MODEL, [b_values, bVec_original]]#
                        signal_model_parameters.append(image_orientation_patient)
                        print("signal_model_parameters")
                        print(signal_model_parameters[2])
            
                        ## read elastix params
                        elastixImageFilter = sitk.ElastixImageFilter()
                        elastix_model_parameters = elastixImageFilter.ReadParameterFile("/Users/kanishkasharma/Documents/GitHub/MDR_Library/Elastix_Parameters_Files/iBEAt/BSplines_DTI.txt")
                        elastix_model_parameters['MaximumNumberOfIterations'] = ['256'] 
                        elastixImageFilter.SetParameterMap(elastix_model_parameters) 
                        elastixImageFilter.PrintParameterMap()
                      
                        output_dir =  OUTPUT_REG_PATH + '/DTI/' ##
                        ## full image analysis
                        shape = np.shape(original_images)
         
                        original_images_reslice = np.zeros([shape[0]*shape[1], shape[2]], dtype=np.uint16) #

                        for i in range(shape[2]):#dynamics
                            reslice = sitk.GetImageFromArray(original_images[:,:,i])
                            reslice.SetOrigin(slice_parameters[0])
                            reslice.SetSpacing(slice_parameters[1])
                            reslice.__SetPixelAsUInt16__
                            original_images_reslice[:,i]  = reslice

                        original_images_reslice = np.reshape(original_images_reslice, [shape[0], shape[1], shape[2]]) # required as the above GetImageFromArray reshapes the source to (147456,)
                        print("original_images image shape")
                        print(np.shape(original_images_reslice))

                        im_original = np.zeros([shape[0],shape[1]], dtype=np.uint16) 

                        for i in range(shape[2]):
                            im_original = Image.fromarray(original_images_reslice[:,:,i])
                            im_original.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/DTI/original/original_series_'+ str(i) + ".tiff")

                        ## uncomment below for full image analysis                 
                        #motion_corrected_images, fit, deformation_field, fitted_quantitative_Params, diagnostics = model_driven_registration(original_images, slice_parameters, signal_model_parameters, elastix_model_parameters, precision  = 0.5)
                        
                        # plt.imshow(original_images[:,:,47].squeeze(), cmap="gray")
                        # plt.colorbar()
                        # plt.show()

                        ## cropped image analysis; ## 1mm precision as pixel size is 1.04mm
                        ## TODO: comment line below for full image analysis
                        ## correct the pixel resolution for cropped region
                        
                        cropped_images_reslice = np.zeros([shape_image_arr[0]*shape_image_arr[1], shape_image_arr[2]], dtype=np.uint16) #

                        for i in range(shape_image_arr[2]):#dynamics
                            reslice = sitk.GetImageFromArray(image_arr[:,:,i])
                            reslice.SetOrigin(slice_parameters[0])
                            reslice.SetSpacing(slice_parameters[1])
                            reslice.__SetPixelAsUInt16__
                            cropped_images_reslice[:,i]  = reslice

                        cropped_images_reslice = np.reshape(cropped_images_reslice, [shape_image_arr[0], shape_image_arr[1], shape_image_arr[2]]) # required as the above GetImageFromArray reshapes the source to (147456,)
                        print("cropped image shape")
                        print(np.shape(cropped_images_reslice))

                        im_crop = np.zeros([shape_image_arr[0],shape_image_arr[1]], dtype=np.uint16) 

                        for i in range(shape_image_arr[2]):
                            im_crop = Image.fromarray(cropped_images_reslice[:,:,i])
                            im_crop.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/DTI/original_data_cropped/original_cropped_resliced_'+ str(i) + ".tiff")
        
                        #sys.exit()
                        motion_corrected_images, fit, deformation_field, fitted_quantitative_Params, diagnostics = model_driven_registration(cropped_images_reslice, slice_parameters, signal_model_parameters, elastix_model_parameters, precision  = 1) 

                        diagnostics.to_csv('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/DTI/largest_deformations.csv' , index=False)
                          
                        im_motion_corrected = np.zeros([shape[0],shape[1]], dtype=np.uint16) #
                        im_fit = np.zeros([shape[0],shape[1]], dtype=np.uint16) #
                        im_deform_x = np.zeros([shape[0],shape[1],2], dtype=np.uint16)
                        im_deform_y = np.zeros([shape[0],shape[1],2], dtype=np.uint16)

                        for i in range(shape[2]):
                            im_motion_corrected = Image.fromarray(motion_corrected_images[:,:,i])
                            im_fit = Image.fromarray(fit[:,:,i])
                            im_deform_x = Image.fromarray(deformation_field[:,:,0,i])
                            im_deform_y = Image.fromarray(deformation_field[:,:,1,i])
                            im_motion_corrected.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/DTI/coregistered/co-registered_test_'+ str(i) + ".tiff")
                            im_fit.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/DTI/fit/final_fit_'+ str(i) + ".tiff")
                            im_deform_x.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/DTI/final_deformation_x/final_deformation_x_' + str(i) + ".tiff")
                            im_deform_y.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/DTI/final_deformation_y/final_deformation_y_' + str(i) + ".tiff")
                        
                      
                        print("fitted_quantitative params shape")
                        print(np.shape(fitted_quantitative_Params)) ## 1-D list

                        ## Fitted Parameters:fitted_parameters 
                        # #every_second_element = a_list[::2] eg
                        FA = np.reshape(fitted_quantitative_Params[2::4],[shape[0],shape[1]])
                        FA_Img = Image.fromarray(FA)
                        FA_Img.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/DTI/fitted_params/FA' + ".tiff")
                        
                        ADC = np.reshape(fitted_quantitative_Params[3::4],[shape[0],shape[1]])
                        ADC_Img = Image.fromarray(ADC)
                        ADC_Img.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/DTI/fitted_params/ADC' + ".tiff")
                        
                        plt.imshow(FA)
                        plt.colorbar()
                        plt.show()  

                        ## visualise results
                        plt.imshow(original_images[:,:,47].squeeze(), cmap="gray")
                        plt.colorbar()
                        plt.show()  
                        
                        plt.imshow(motion_corrected_images[:,:,47].squeeze(), cmap="gray")
                        plt.colorbar()
                        plt.show() 
                        

                    elif sequence == 'IVIM': # for mono-exponential decay this is just DWI


                        ## read sequence related parameters
                        b_values, bVec_original, image_orientation_patient, slice_sorted_b_values  = iBEAt_IVIM_monoexponential.read_dicom_tags_IVIM(fname, lstFilesDCM)
                        print("b_values")
                        print(b_values)
                        print("bVec_original")
                        print(bVec_original)
                        print("image_orientation_patient")
                        print(image_orientation_patient) # array has repititions so take only 1st set of 6
                       # sys.exit()
                       
                        for i, s in enumerate(slice_sorted_acq_time):
                            img2d = s.pixel_array
                            original_images[:, :, i] = img2d
                            resaved = Image.fromarray(img2d)
                            ## TODO change according to series
                            resaved.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/IVIM/original_data_resaved/original_resaved_'+ str(i) + ".tiff")

                         ## crop image for test (Y,X)
                        image_arr = original_images[20:130, 20:160, :] # 
                        shape_image_arr = np.shape(image_arr)
 
                        MODEL = [iBEAt_IVIM_monoexponential,'fitting'] #filename and function name
                        
                        signal_model_parameters = [MODEL, [b_values, bVec_original]]#, [image_orientation_patient]]
                        signal_model_parameters.append(image_orientation_patient)
                        print("signal_model_parameters")
                        print(signal_model_parameters[1][0])
                        #sys.exit()
                        ## read elastix params
                        elastixImageFilter = sitk.ElastixImageFilter()
                        elastix_model_parameters = elastixImageFilter.ReadParameterFile("/Users/kanishkasharma/Documents/GitHub/MDR_Library/Elastix_Parameters_Files/iBEAt/BSplines_IVIM.txt")
                        elastix_model_parameters['MaximumNumberOfIterations'] = ['256'] #
                        elastixImageFilter.SetParameterMap(elastix_model_parameters) 
                        elastixImageFilter.PrintParameterMap()
                      
                        output_dir =  OUTPUT_REG_PATH + '/IVIM/' 

                        ## full image analysis
                        shape = np.shape(original_images)

                        original_images_reslice = np.zeros([shape[0]*shape[1], shape[2]], dtype=np.uint16) #

                        for i in range(shape[2]):#dynamics
                            reslice = sitk.GetImageFromArray(original_images[:,:,i])
                            reslice.SetOrigin(slice_parameters[0])
                            reslice.SetSpacing(slice_parameters[1])
                            reslice.__SetPixelAsUInt16__
                            original_images_reslice[:,i]  = reslice

                        original_images_reslice = np.reshape(original_images_reslice, [shape[0], shape[1], shape[2]]) # 
                        print("original_images image shape")
                        print(np.shape(original_images_reslice))

                        im_original = np.zeros([shape[0],shape[1]], dtype=np.uint16) 

                        for i in range(shape[2]):
                            im_original = Image.fromarray(original_images_reslice[:,:,i])
                            im_original.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/IVIM/original/original_series_'+ str(i) + ".tiff")

                        ## uncomment below for full image analysis                 
                        #motion_corrected_images, fit, deformation_field, fitted_quantitative_Params, diagnostics = model_driven_registration(original_images, slice_parameters, signal_model_parameters, elastix_model_parameters, precision  = 0.5)
                        
                        # plt.imshow(original_images[:,:,47].squeeze(), cmap="gray")
                        # plt.colorbar()
                        # plt.show()
                        
                        cropped_images_reslice = np.zeros([shape_image_arr[0]*shape_image_arr[1], shape_image_arr[2]], dtype=np.uint16) #

                        for i in range(shape_image_arr[2]):#dynamics
                            reslice = sitk.GetImageFromArray(image_arr[:,:,i])
                            reslice.SetOrigin(slice_parameters[0])
                            reslice.SetSpacing(slice_parameters[1])
                            reslice.__SetPixelAsUInt16__
                            cropped_images_reslice[:,i]  = reslice

                        cropped_images_reslice = np.reshape(cropped_images_reslice, [shape_image_arr[0], shape_image_arr[1], shape_image_arr[2]]) # required as the above GetImageFromArray reshapes the source to (147456,)
                        print("cropped image shape")
                        print(np.shape(cropped_images_reslice))

                        im_crop = np.zeros([shape_image_arr[0],shape_image_arr[1]], dtype=np.uint16) 

                        for i in range(shape_image_arr[2]):
                            im_crop = Image.fromarray(cropped_images_reslice[:,:,i])
                            im_crop.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/IVIM/original_data_cropped/original_cropped_resliced_'+ str(i) + ".tiff")
        
                        motion_corrected_images, fit, deformation_field, fitted_quantitative_Params, diagnostics = model_driven_registration(cropped_images_reslice, slice_parameters, signal_model_parameters, elastix_model_parameters, precision  = 1) 

                        diagnostics.to_csv('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/IVIM/largest_deformations.csv' , index=False)
                          
                        im_motion_corrected = np.zeros([shape[0],shape[1]], dtype=np.uint16) #
                        im_fit = np.zeros([shape[0],shape[1]], dtype=np.uint16) #
                        im_deform_x = np.zeros([shape[0],shape[1],2], dtype=np.uint16)
                        im_deform_y = np.zeros([shape[0],shape[1],2], dtype=np.uint16)

                        for i in range(shape[2]):
                            im_motion_corrected = Image.fromarray(motion_corrected_images[:,:,i])
                            im_fit = Image.fromarray(fit[:,:,i])
                            im_deform_x = Image.fromarray(deformation_field[:,:,0,i])
                            im_deform_y = Image.fromarray(deformation_field[:,:,1,i])
                            im_motion_corrected.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/IVIM/coregistered/co-registered_test_'+ str(i) + ".tiff")
                            im_fit.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/IVIM/fit/final_fit_'+ str(i) + ".tiff")
                            im_deform_x.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/IVIM/final_deformation_x/final_deformation_x_' + str(i) + ".tiff")
                            im_deform_y.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/IVIM/final_deformation_y/final_deformation_y_' + str(i) + ".tiff")
                        
                      
                        print("fitted_quantitative params shape")
                        print(np.shape(fitted_quantitative_Params)) ## 1-D list

                        S0_IVIM_monopexp = np.reshape(fitted_quantitative_Params[::2],[shape_image_arr[0],shape_image_arr[1]]) ## every 2nd element starting 0 from param list
                        S0_IVIM_monoexp_Img = Image.fromarray(S0_IVIM_monopexp)
                        S0_IVIM_monoexp_Img.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/IVIM/fitted_params/S0_IVIM_monoexp' + ".tiff")
                    
                        ADC_IVIM_monopexp = np.reshape(fitted_quantitative_Params[1::2],[shape_image_arr[0],shape_image_arr[1]]) ## every 2nd element starting 1 from param list
                        ADC_IVIM_monopexp_Img = Image.fromarray(ADC_IVIM_monopexp)
                        ADC_IVIM_monopexp_Img.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/IVIM/fitted_params/ADC_IVIM_monopexp' + ".tiff")
                    
                        plt.imshow(ADC_IVIM_monopexp, extent =[0, 0.005, 0, 0.005])
                        plt.colorbar()
                        plt.show()  

                        ## visualise results
                        plt.imshow(original_images[:,:,5].squeeze(), cmap="gray")
                        plt.colorbar()
                        plt.show()  
                        
                        plt.imshow(motion_corrected_images[:,:,5].squeeze(), cmap="gray")
                        plt.colorbar()
                        plt.show() 


                    elif sequence == 'DCE':

                        for i, s in enumerate(slice_sorted_acq_time):
                            img2d = s.pixel_array
                            original_images[:, :, i] = img2d
                            resaved = Image.fromarray(img2d)
                            ## TODO change according to series
                            resaved.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/DCE/original_data_resaved/original_resaved_'+ str(i) + ".tiff")
                       
                         ## crop image for test
                        image_arr = original_images[80:272, 32:224, :] #
                        #original_images[172:192, 192:212, :]## test size
                        shape_image_arr = np.shape(image_arr)
                        # plt.imshow(image_arr[:,:,47].squeeze(), cmap="gray")
                        # plt.colorbar()
                        # plt.show()
                        # im_crop = np.zeros([shape_image_arr[0],shape_image_arr[1]], dtype=np.uint16) 

                        ## read the sequence related params
                        aif, times = iBEAt_DCE.load_txt(AIFs_PATH + '/' + str(patient_folder) + '/' + 'AIF__2C Filtration__Curve.txt')
                        aif.append(aif[-1])
                        times.append(times[-1])

                        MODEL = [iBEAt_DCE,'fitting'] #filename and function name
                        
                        signal_model_parameters = [MODEL, aif, times]
        
                        ## read elastix params
                        elastixImageFilter = sitk.ElastixImageFilter()
                        elastix_model_parameters = elastixImageFilter.ReadParameterFile("/Users/kanishkasharma/Documents/GitHub/MDR_Library/Elastix_Parameters_Files/iBEAt/BSplines_DCE.txt")
                        elastix_model_parameters['MaximumNumberOfIterations'] = ['256'] #
                        elastixImageFilter.SetParameterMap(elastix_model_parameters) 
                        elastixImageFilter.PrintParameterMap()
                      
                        output_dir =  OUTPUT_REG_PATH + '/DCE/'
   
                        ## full image analysis
                        shape = np.shape(original_images)
  
                        original_images_reslice = np.zeros([shape[0]*shape[1], shape[2]], dtype=np.uint16) #

                        for i in range(shape[2]):#dynamics
                            reslice = sitk.GetImageFromArray(original_images[:,:,i])
                            reslice.SetOrigin(slice_parameters[0])
                            reslice.SetSpacing(slice_parameters[1])
                            reslice.__SetPixelAsUInt16__
                            original_images_reslice[:,i]  = reslice

                        original_images_reslice = np.reshape(original_images_reslice, [shape[0], shape[1], shape[2]]) # 
                        print("original_images image shape")
                        print(np.shape(original_images_reslice))

                        im_original = np.zeros([shape[0],shape[1]], dtype=np.uint16) 

                        for i in range(shape[2]):
                            im_original = Image.fromarray(original_images_reslice[:,:,i])
                            im_original.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/DCE/original/original_series_'+ str(i) + ".tiff")

                        ## uncomment below for full image analysis                 
                        #motion_corrected_images, fit, deformation_field, fitted_quantitative_Params, diagnostics = model_driven_registration(original_images_reslice, slice_parameters, signal_model_parameters, elastix_model_parameters, precision  = 0.5)
                        
                        # plt.imshow(original_images[:,:,47].squeeze(), cmap="gray")
                        # plt.colorbar()
                        # plt.show()

                        ## cropped image analysis; ## 1mm precision 
                        ## TODO: comment line below for full image analysis
                        cropped_images_reslice = np.zeros([shape_image_arr[0]*shape_image_arr[1], shape_image_arr[2]], dtype=np.uint16) #

                        for i in range(shape_image_arr[2]):#dynamics
                            reslice = sitk.GetImageFromArray(image_arr[:,:,i])
                            reslice.SetOrigin(slice_parameters[0])
                            reslice.SetSpacing(slice_parameters[1])
                            reslice.__SetPixelAsUInt16__
                            cropped_images_reslice[:,i]  = reslice

                        cropped_images_reslice = np.reshape(cropped_images_reslice, [shape_image_arr[0], shape_image_arr[1], shape_image_arr[2]]) # required as the above GetImageFromArray reshapes the source to (147456,)
                        print("cropped image shape")
                        print(np.shape(cropped_images_reslice))

                        im_crop = np.zeros([shape_image_arr[0],shape_image_arr[1]], dtype=np.uint16) 

                        for i in range(shape_image_arr[2]):
                            im_crop = Image.fromarray(cropped_images_reslice[:,:,i])
                            im_crop.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/DCE/original_data_cropped/original_cropped_'+ str(i) + ".tiff")
        
                        motion_corrected_images, fit, deformation_field, fitted_quantitative_Params, diagnostics = model_driven_registration(cropped_images_reslice, slice_parameters, signal_model_parameters, elastix_model_parameters, precision  = 1) 
            
                        im_motion_corrected = np.zeros([shape[0],shape[1]], dtype=np.uint16) #
                        im_fit = np.zeros([shape[0],shape[1]], dtype=np.uint16) #
                        im_deform_x = np.zeros([shape[0],shape[1],2], dtype=np.uint16)
                        im_deform_y = np.zeros([shape[0],shape[1],2], dtype=np.uint16)

                        for i in range(shape[2]):
                            im_motion_corrected = Image.fromarray(motion_corrected_images[:,:,i])
                            im_fit = Image.fromarray(fit[:,:,i])
                            im_deform_x = Image.fromarray(deformation_field[:,:,0,i])
                            im_deform_y = Image.fromarray(deformation_field[:,:,1,i])
                            im_motion_corrected.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/DCE/coregistered/co-registered_test_'+ str(i) + ".tiff")
                            im_fit.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/DCE/fit/final_fit_'+ str(i) + ".tiff")
                            im_deform_x.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/DCE/final_deformation_x_/final_deformation_x_' + str(i) + ".tiff")
                            im_deform_y.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/DCE/final_deformation_y_/final_deformation_y_' + str(i) + ".tiff")
                        
                        diagnostics.to_csv('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/DCE/largest_deformations.csv' , index=False)
                        
                        print("fitted_quantitative params shape")
                        print(np.shape(fitted_quantitative_Params)) ## 1-D list

                        ## Fitted Parameters:fitted_parameters = [Fp, Tp, Ps, Te] 
                        Fp = np.reshape(fitted_quantitative_Params[::4],[shape_image_arr[0],shape_image_arr[1]]) 
                        Fp_Img = Image.fromarray(Fp)
                        Fp_Img.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/DCE/fitted_params/Fp' + ".tiff")
                        
                        Tp = np.reshape(fitted_quantitative_Params[1::4],[shape_image_arr[0],shape_image_arr[1]]) 
                        Tp_Img = Image.fromarray(Tp)
                        Tp_Img.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/DCE/fitted_params/Tp' + ".tiff")
                       
                        Ps = np.reshape(fitted_quantitative_Params[2::4],[shape_image_arr[0],shape_image_arr[1]])
                        Ps_Img = Image.fromarray(Ps)
                        Ps_Img.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/DCE/fitted_params/Ps' + ".tiff")
                        
                        Te = np.reshape(fitted_quantitative_Params[3::4],[shape_image_arr[0],shape_image_arr[1]])
                        Te_Img = Image.fromarray(Te)
                        Te_Img.save('/Users/kanishkasharma/Documents/GitHub/iBEAt-Library/within_sequence_registration_iBEAt/iBEAt_MDR/output/final/DCE/fitted_params/Te' + ".tiff")
                        
                        plt.imshow(Fp)
                        plt.colorbar()
                        plt.show()  

                        ## visualise results
                        plt.imshow(original_images[:,:,47].squeeze(), cmap="gray")
                        plt.colorbar()
                        plt.show()  
                        
                        plt.imshow(motion_corrected_images[:,:,47].squeeze(), cmap="gray")
                        plt.colorbar()
                        plt.show() 


                    else:
                        raise Exception("iBEAt sequence not recognised")





    
