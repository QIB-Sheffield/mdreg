"""
MODEL DRIVEN REGISTRATION for iBEAt study: quantitative renal MRI
@Kanishka Sharma 2021
Test script for DWI sequence using Model driven registration Library
"""
import sys
import glob
import os
import numpy as np
import pydicom
from pathlib import Path 
import importlib
import time
from itertools import repeat
from MDR.MDR import model_driven_registration  
from MDR.Tools import (read_DICOM_files, get_sitk_image_details_from_DICOM, 
                      sort_all_slice_files_acquisition_time, read_elastix_model_parameters,
                      export_images, export_maps)


np.set_printoptions(threshold=sys.maxsize)

def main():
    # selected sequence to process
    sequence = 'IVIM'
    # number of expected slices to process (example: iBEAt study number of slice = 30)
    slices = 30    
    # path definition  
    # your 'os.getcwd()' path should point to your local directory containing the  MDR-Library 
    # eg: /Users/kanishkasharma/Documents/GitHub/MDR_Library
    print(os.getcwd()) 

    DATA_PATH = os.getcwd() + r'/tests/test_data/DICOMs'
    OUTPUT_REG_PATH = os.getcwd() + r'/MDR_registration_output'
    Elastix_Parameter_file_PATH = os.getcwd() + r'/Elastix_Parameters_Files/iBEAt/BSplines_IVIM.txt' 
    output_dir =  OUTPUT_REG_PATH + '/DWI/'

    # Organize files per each sequence:
    os.chdir(DATA_PATH)    
    # list all patient folders available to be processed
    patients_folders = os.listdir()
    # select patient folder to be processed from the list of available patient DICOMs supplied in patients_folders
    for patient_folder in patients_folders:
        if patient_folder not in ['test_case_iBEAt_4128009']: # eg: test case selected to be processed - change to your own test case
            continue
        # read path to the sequence to be processed for selected test patient case: eg: DWI
        sequence_images_path = patient_folder + '/' + str(sequence) + '/DICOM'
        os.chdir(DATA_PATH + '/' + sequence_images_path)
        # read all dicom files for selected sequence
        dcm_files_found = glob.glob("*.dcm")
        if not dcm_files_found:
            dcm_files_found = glob.glob("*.IMA") # if sequence is IMA format instead of dcm
        # slice to be processed from selected sequence
        for slice in range(1, slices+1):
            current_slice = sequence + '_slice_' + str(slice)
            # single slice processing for DWI sequence (here selected slice number is 15)
            if current_slice not in [sequence + '_slice_15']:
                continue
            # read slice path to be processed
            slice_path = DATA_PATH + '/' + sequence_images_path + '/' + current_slice
            data = Path(slice_path)
            # list of all DICOMs to be processed for the selected slice (example: slice number = 15 here)
            lstFilesDCM = list(data.glob('**/*.IMA')) 
    
            # read all dicom files for the selected sequence and slice
            files, ArrayDicomiBEAt, filenameDCM = read_DICOM_files(lstFilesDCM)
            # get sitk image parameters for registration (pixel spacing)
            image_parameters = get_sitk_image_details_from_DICOM(slice_path)
            # sort slices correctly - based on acquisition time for model driven registration
            sorted_slice_files = sort_all_slice_files_acquisition_time(files)
            # run DWI MDR test 
            iBEAt_test_DWI(Elastix_Parameter_file_PATH, output_dir, sorted_slice_files, ArrayDicomiBEAt, image_parameters, filenameDCM, lstFilesDCM)


# test DWI using model driven registration
def iBEAt_test_DWI(Elastix_Parameter_file_PATH, output_dir, sorted_slice_files, ArrayDicomiBEAt, image_parameters, filenameDCM, lstFilesDCM):
    """ Example application of MDR in renal DWI (iBEAt data).  

    Description
    -----------
    This function performs model driven registration for selected DWI sequence on a single selected slice 
    and returns as output the MDR registered images, signal model fit, deformation field x, deformation field y, 
    fitted parameters S0 and ADC, and the final diagnostics.  
    
    Args
    ----
    Elastix_Parameter_file_PATH (string): complete path to the elastix parameter file to be used.  
    output_dir (string): directory where results are saved.  
    slice_sorted_files (list): selected slices to process using MDR: sorted according to acquisition time.   
    ArrayDicomiBEAt (numpy.ndarray): input DICOM to numpy array (unsorted).  
    image_parameters (SITK input): image spacing.  
    filenameDCM (pathlib.PosixPath): dicom filenames to process.  
    lstFilesDCM (list): list of dicom files to process.  
    """
    start_computation_time = time.time()
    # define numpy array with same input shape as original DICOMs
    image_shape = np.shape(ArrayDicomiBEAt)
    original_images = np.zeros(image_shape)

    # initialise original_images with sorted acquisiton times to run MDR
    for i, s in enumerate(sorted_slice_files):
        img2d = s.pixel_array
        original_images[:, :, i] = img2d

    # read module filename for DWI
    full_module_name = "models.DWI_monoexponential"
    # generate a module named as a string
    model = importlib.import_module(full_module_name)
    # read signal model parameters
    signal_model_parameters = read_signal_model_parameters(filenameDCM, lstFilesDCM)
    # read elastix model parameters
    elastix_model_parameters = read_elastix_model_parameters(Elastix_Parameter_file_PATH, ['MaximumNumberOfIterations', 256])
    
    #Perform MDR
    MDR_output = model_driven_registration(original_images, image_parameters, model, signal_model_parameters, elastix_model_parameters, precision = 1, function = 'main')

    #Export results
    export_images(MDR_output[0], output_dir +'/coregistered/MDR-registered_DWI_')
    export_images(MDR_output[1], output_dir +'/fit/fit_image_')
    export_images(MDR_output[2][:,:,0,:], output_dir +'/deformation_field/final_deformation_x_')
    export_images(MDR_output[2][:,:,1,:], output_dir +'/deformation_field/final_deformation_y_')
    export_maps(MDR_output[3][0,:], output_dir + '/fitted_parameters/S0', np.shape(original_images))
    export_maps(MDR_output[3][1,:], output_dir + '/fitted_parameters/ADC', np.shape(original_images))
    MDR_output[4].to_csv(output_dir + 'DWI_largest_deformations.csv')

    # Report computation times
    end_computation_time = time.time()
    print("total computation time for MDR (minutes taken:)...")
    print(0.0166667*(end_computation_time - start_computation_time)) # in minutes
    print("completed MDR registration!")
    print("Finished processing Model Driven Registration case for iBEAt study DWI sequence!")


## read sequence acquisition parameter for signal modelling
def read_signal_model_parameters(filenameDCM, lstFilesDCM):

    b_values, bVec_original, image_orientation_patient, slice_sorted_b_values  = read_dicom_tags_IVIM(filenameDCM, lstFilesDCM)
    # select signal model paramters
    signal_model_parameters = []
    signal_model_parameters.append(b_values)
    signal_model_parameters.append(bVec_original)
    signal_model_parameters.append(image_orientation_patient)

    return signal_model_parameters

def read_dicom_tags_IVIM(fname,lstFilesDCM):
    """ This function reads the DICOM tags from the IVIM sequence and returns the corresponding DWI/IVIM tags.  

    Args
    ----
    filenameDCM (pathlib.PosixPath): dicom filenames to process.  
    lstFilesDCM (list): list of dicom files to process.  

    Returns
    -------
    b-values (list): list of DWI/IVIM b-values (s/mm2).   
    b_Vec_original (list): original b-vectors as list.  
    image_orientation_patient (list):  patient orientation as list.  
    slice_sorted_b_values (list): list of slices sorted according to b-values.  
    """

    b_values = []
    b_values_sort = []
    b_Vec_original = []
    image_orientation_patient = []
    files = []
 
    g_dir_01 = [1,0,0]
    g_dir_02 = [0,1,0]
    g_dir_03 = [0,0,1]
    #TODO: to check gradient files
    for i in range(10):
        b_Vec_original.append(g_dir_01)

    for i in range(10,20):
        b_Vec_original.append(g_dir_02)

    for i in range(21,30):
        b_Vec_original.append(g_dir_03)

    ## b-values not avaailable in DICOM header
    b_values = [0,10.000086, 19.99908294, 30.00085926, 50.00168544, 80.007135, 100.0008375, 199.9998135, 300.0027313, 600.0]
    b_values = list(repeat(b_values, 3)) # repeated 3 times for 3 sets of IVIM images

    for fname in lstFilesDCM:
        dataset = pydicom.dcmread(fname)
      
        image_orientation_patient.append(dataset.ImageOrientationPatient)
        b_values_sort.append(dataset[0x19, 0x100c].value)
        files.append(pydicom.dcmread(fname)) 
 
    sort_index = np.argsort(b_values_sort) 
    
    b_values_sort.sort()
   
    slice_b_values= []
    slice_sorted_b_values= []

    for f in files: 
        slice_b_values.append(f)

    # sorted slices using sorted b values
    for i in range(0, len(slice_b_values)):
         slice_sorted_b_values.append(slice_b_values[sort_index[i]])
   
    return b_values, b_Vec_original, image_orientation_patient, slice_sorted_b_values 