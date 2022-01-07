"""
MODEL DRIVEN REGISTRATION (MDR) for quantitative renal MRI  
@Kanishka Sharma 2021  
Test script for T2 mapping within-sequence registration using the MDR Library  
"""
import sys
import glob
import os
import numpy as np
import importlib
from pathlib import Path 
import time
from MDR.MDR import model_driven_registration
from MDR.Tools import (read_DICOM_files, get_sitk_image_details_from_DICOM, 
                      sort_all_slice_files_acquisition_time, read_elastix_model_parameters,
                      export_images, export_maps)
np.set_printoptions(threshold=sys.maxsize)


def main():
    # selected sequence to process
    sequence = 'T2'
    # number of expected slices to process (example: iBEAt study number of slice = 5)
    slices = 5    
    # path definition: your 'os.getcwd()' path should point to your local directory containing the MDR-Library 
    # eg: /Users/kanishkasharma/Documents/GitHub/MDR_Library
    print(os.getcwd()) 

    DATA_PATH = os.getcwd() + r'/tests/test_data/DICOMs'
    OUTPUT_REG_PATH = os.getcwd() + r'/MDR_registration_output'
    Elastix_Parameter_file_PATH = os.getcwd() + r'/Elastix_Parameters_Files/iBEAt/BSplines_T2.txt' 
    output_dir =  OUTPUT_REG_PATH + '/T2/'

    # Organize files per each sequence:
    os.chdir(DATA_PATH)    
    # list all patient folders available to be processed
    patients_folders = os.listdir()
    # select patient folder to be processed from the list of available patient DICOMs supplied in patients_folders
    for patient_folder in patients_folders:
        if patient_folder not in ['test_case_iBEAt_4128009']: # eg: test case selected to be processed - change to your own test case
            continue
        # read path to the sequence to be processed for selected test patient case: eg: T2
        sequence_images_path = patient_folder + '/' + str(sequence) + '/DICOM'
        os.chdir(DATA_PATH + '/' + sequence_images_path)
        # read all dicom files for selected sequence
        dcm_files_found = glob.glob("*.dcm")
        if not dcm_files_found:
            dcm_files_found = glob.glob("*.IMA") # if sequence is IMA format instead of dcm
        # slice to be processed from selected sequence
        for slice in range(1, slices+1):
            current_slice = sequence + '_slice_' + str(slice)
            # single slice processing for T2* sequence (here selected slice number is 3)
            if current_slice not in [sequence + '_slice_3']:
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
            # run T2 star MDR test function
            iBEAt_test_T2(Elastix_Parameter_file_PATH, output_dir, sorted_slice_files, ArrayDicomiBEAt, image_parameters)

                    
def iBEAt_test_T2(Elastix_Parameter_file_PATH, output_dir, sorted_slice_files, ArrayDicomiBEAt, image_parameters):
    """ Example application of MDR in renal T2 mapping (iBEAt study MR data).
     
    Description
    -----------
    This function performs model driven registration for selected T2-mapping sequence on a single selected slice 
    and returns as output the MDR registered images, signal model fit, deformation field x, deformation field y,
    fitted parameters S0 and T2 map, and the final diagnostics.

    Args
    ----
    Elastix_Parameter_file_PATH (string): complete path to the Elastix parameter file to be used
    output_dir (string): directory where results are saved  
    slice_sorted_files (list): selected slices to process using MDR - sorted according to acquisition time   
    ArrayDicomiBEAt (numpy.ndarray): input DICOMs to numpy array (unsorted)  
    image_parameters (SITK tuple): distance between pixels (in mm) along each dimension.  

    """
    #start MDR computation time
    start_computation_time = time.time()
    # define numpy array with same input shape as original DICOMs
    image_shape = np.shape(ArrayDicomiBEAt)
    original_images = np.zeros(image_shape)

    # initialise original_images with sorted acquisiton times to run MDR
    for i, s in enumerate(sorted_slice_files):
        img2d = s.pixel_array
        original_images[:, :, i] = img2d
    
    # generate a module named as a string
    full_module_name = "models.T2"
    # import the module for signal model fit
    model = importlib.import_module(full_module_name) 
    # read signal model input parameters
    signal_model_parameters = read_signal_model_parameters()
    # read elastix model parameters 
    elastix_model_parameters = read_elastix_model_parameters(Elastix_Parameter_file_PATH, ['MaximumNumberOfIterations', 256])
    #Perform MDR
    MDR_output = model_driven_registration(original_images, image_parameters, model, signal_model_parameters, elastix_model_parameters, precision = 1, function = 'main')

    #Export results
    export_images(MDR_output[0], output_dir +'/coregistered/MDR-registered_T2_')
    export_images(MDR_output[1], output_dir +'/fit/fit_image_')
    export_images(MDR_output[2][:,:,0,:], output_dir +'/deformation_field/final_deformation_x_')
    export_images(MDR_output[2][:,:,1,:], output_dir +'/deformation_field/final_deformation_y_')
    export_maps(MDR_output[3][0,:], output_dir + '/fitted_parameters/S0', np.shape(original_images))
    export_maps(MDR_output[3][1,:], output_dir + '/fitted_parameters/T2map', np.shape(original_images))
    MDR_output[4].to_csv(output_dir + 'T2_largest_deformations.csv')

    # Report computation times
    end_computation_time = time.time()
    print("total computation time for MDR (minutes taken:)...")
    print(0.0166667*(end_computation_time - start_computation_time)) # in minutes
    print("completed MDR!")
    print("Finished processing Model Driven Registration (MDR) case for iBEAt study T2-mapping sequence!")

 
 ## read sequence acquisition parameter for signal modelling
def read_signal_model_parameters(): 
   
    T2_prep_times = read_prep_times()

    return T2_prep_times


def read_prep_times():
    """ This function manually reads T2-prep times (iBEAt study specific input paramter currently 
    not available in the DICOM header) and returns it as a list."""
    ## hard coded as these are not available in the anonymised Siemens dicom tags
    T2_prep_times = [0,30,40,50,60,70,80,90,100,110,120]
    
    return T2_prep_times
