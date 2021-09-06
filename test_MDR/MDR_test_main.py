"""
MODEL DRIVEN REGISTRATION for iBEAt study: quantitative renal MRI
@Kanishka Sharma 2021
Main script to test Model driven registration Library
Test case MR sequences from the iBEAt study are provided 
"""
import sys
import glob
import os
import numpy as np
import SimpleITK as sitk
import pydicom
from collections import OrderedDict
from pathlib import Path 
from test_MDR import MDR_test_T2, MDR_test_T2star, MDR_test_DTI, MDR_test_DWI, MDR_test_DCE

np.set_printoptions(threshold=sys.maxsize)


if __name__ == '__main__':
  
    ## select sequence to process
    SEQUENCES = ['T2', 'T2star', 'DTI', 'IVIM', 'DCE']
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

                    image_parameters = []

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
                     
                    image_parameters = [origin, spacing]

                    img_shape = np.shape(ArrayDicomiBEAt)

## WHY ARE THESE SET TO ZERO?

                    original_images = np.zeros(img_shape)
  
                    ## MDR for T2-mapping sequence
                    if sequence == 'T2':

                        ## output directory for final registration results 
                        output_dir =  OUTPUT_REG_PATH + '/T2/'

                        MDR_test_T2.iBEAt_test_T2(Elastix_Parameter_file_PATH, output_dir, slice_sorted_acq_time, original_images, image_parameters)

                    # MDR for T2star-mapping sequence
                    elif sequence == 'T2star':
                       
                        ## output directory for final registration results 
                        output_dir =  OUTPUT_REG_PATH + '/T2star/'
                       
                        MDR_test_T2star.iBEAt_test_T2star(Elastix_Parameter_file_PATH, output_dir, slice_sorted_acq_time, original_images, image_parameters, fname, lstFilesDCM)
  
                    ## MDR for DTI sequence
                    elif sequence == 'DTI':

                        ## output directory for final registration results 
                        output_dir =  OUTPUT_REG_PATH + '/DTI/'

                        MDR_test_DTI.iBEAt_test_DTI(Elastix_Parameter_file_PATH, output_dir, slice_sorted_acq_time, original_images, image_parameters, fname, lstFilesDCM)

                    ## MDR for IVIM/DWI sequence
                    elif sequence == 'IVIM':

                        ## output directory for final registration results 
                        output_dir =  OUTPUT_REG_PATH + '/DWI/'

                        MDR_test_DWI.iBEAt_test_DWI(Elastix_Parameter_file_PATH, output_dir, slice_sorted_acq_time, original_images, image_parameters, fname, lstFilesDCM)

                    ## MDR for DCE sequence
                    elif sequence == 'DCE':

                        ## output directory for final registration results 
                        output_dir =  OUTPUT_REG_PATH + '/DCE/'

                        MDR_test_DCE.iBEAt_test_DCE(Elastix_Parameter_file_PATH, output_dir, slice_sorted_acq_time, original_images, image_parameters, AIFs_PATH, patient_folder)


                    else:
                        raise Exception("iBEAt sequence not recognised")





    
