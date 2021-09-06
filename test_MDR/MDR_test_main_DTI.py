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
from test_MDR import MDR_test_DTI

np.set_printoptions(threshold=sys.maxsize)


def main():
  
    sequence = 'DTI'
    num_of_files = 4380
    slices = 30    

    ## path definitions   
    ## your 'os.getcwd()' path should point to your local MDR_Library directory
    print(os.getcwd()) # eg: /Users/kanishkasharma/Documents/GitHub/MDR_Library

# Kanishka, make a separate datafolder. Normally you wouldn't save the data in the same place as the code

    DATA_PATH = os.getcwd() + r'/test_MDR/test_data/DICOMs'
    OUTPUT_REG_PATH = os.getcwd() + r'/test_MDR/iBEAt_MDR/MDR_registration_output'
    Elastix_Parameter_file_PATH = os.getcwd() + r'/Elastix_Parameters_Files/iBEAt' 
    output_dir =  OUTPUT_REG_PATH + '/DTI/'

    # Organize files per each sequence:

    os.chdir(DATA_PATH)    
    patients_folders = os.listdir()
    for patient_folder in patients_folders:
        if patient_folder not in ['test_case_iBEAt_4128009']: 
            continue
        
        sequence_images_path = patient_folder + '/' + str(sequence) + '/DICOM'
        os.chdir(DATA_PATH + '/' + sequence_images_path)
    
        dcm_files_found = glob.glob("*.dcm")
        if not dcm_files_found:
            dcm_files_found = glob.glob("*.IMA")
    
        for slice in range(1, slices+1):
            current_slice = sequence + '_slice_' + str(slice)
            ## single slice processing per sequence for test
            if current_slice not in [sequence + '_slice_15']:
                continue

            slice_path = DATA_PATH + '/' + sequence_images_path + '/' + current_slice
            data = Path(slice_path)
            lstFilesDCM = list(data.glob('**/*.IMA')) 
    
            # KANISHKA: as an illustration - replace this by a more meaninggul name
            # These blocks of code create a very confusing picture as all kinds of
            # temporary variables are created that are not used later anyway
            # So you are leaving your reader to reverse engineer what this is doing
            # Packaging it up in a function  like this makes the whole thing 
            # a lot easier to understand, especially if you choose meaningful names for things
            # Dp this literally everywhere. Your code must become almost a text that you can read line by line
            ArrayDicomiBEAt, files = doing_something(lstFilesDCM)

            # loop through all the DICOM slices
            for filenameDCM in lstFilesDCM:
                ds = pydicom.dcmread(filenameDCM)
                ArrayDicomiBEAt[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array  
            
            image_parameters = get_pixel_details_from_DICOM(slice_path)
            slice_sorted_acq_time = sort_all_images_according_to_the_acquisition_time(files)
            
            img_shape = np.shape(ArrayDicomiBEAt)
            original_images = np.zeros(img_shape) ## Why??????

            ## output directory for final registration results 
            
            MDR_test_DTI.iBEAt_test_DTI(Elastix_Parameter_file_PATH, output_dir, slice_sorted_acq_time, original_images, image_parameters, fname, lstFilesDCM)



def get_pixel_details_from_DICOM(slice_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(slice_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    origin = image.GetOrigin() 
    spacing = image.GetSpacing() 
    return origin, spacing


def sort_all_images_according_to_the_acquisition_time(files):
    slice_sorted_acq_time = []
    skipcount = 0
    for f in files: 
        if hasattr(f, 'AcquisitionTime'):
            slice_sorted_acq_time.append(f)
        else:
            skipcount = skipcount + 1
    print("skipped, no AcquisitionTime: {}".format(skipcount))
    # Is the lamba function really necessary?
    return sorted(slice_sorted_acq_time, key=lambda s: s.AcquisitionTime)  

def doing_something(lstFilesDCM):
    files = []
    for fname in lstFilesDCM:
        files.append(pydicom.dcmread(fname))
    RefDs = pydicom.dcmread(lstFilesDCM[0])
    SeriesPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))           
    ArrayDicomiBEAt = np.zeros(SeriesPixelDims, dtype=RefDs.pixel_array.dtype)
    print("file count: {}".format(len(files)))
    return files, ArrayDicomiBEAt 



    
