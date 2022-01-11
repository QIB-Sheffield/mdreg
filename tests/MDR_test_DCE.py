"""
MODEL DRIVEN REGISTRATION for iBEAt study: quantitative renal MRI  
@Kanishka Sharma  
2021  
Test script for DCE sequence using Model driven registration Library  
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
    sequence = 'DCE'
    # number of expected slices to process (example: iBEAt study number of slice = 9)
    slices = 9    
    # path definition  
    # your 'os.getcwd()' path should point to your local directory containing the  MDR-Library 
    # eg: /Users/kanishkasharma/Documents/GitHub/MDR_Library
    print(os.getcwd()) 

    DATA_PATH = os.getcwd() + r'/tests/test_data/DICOMs'
    AIFs_PATH = os.getcwd() + r'/tests/test_data/AIFs' 
    OUTPUT_REG_PATH = os.getcwd() + r'/MDR_registration_output'
    Elastix_Parameter_file_PATH = os.getcwd() + r'/Elastix_Parameters_Files/iBEAt/BSplines_DCE.txt' 
    output_dir =  OUTPUT_REG_PATH + '/DCE/'

    # Organize files per each sequence:
    os.chdir(DATA_PATH)    
    # list all patient folders available to be processed
    patients_folders = os.listdir()
    # select patient folder to be processed from the list of available patient DICOMs supplied in patients_folders
    for patient_folder in patients_folders:
        if patient_folder not in ['test_case_iBEAt_4128009']: # eg: test case selected to be processed - change to your own test case
            continue
        # read path to the sequence to be processed for selected test patient case: eg: DCE
        sequence_images_path = patient_folder + '/' + str(sequence) + '/DICOM'
        os.chdir(DATA_PATH + '/' + sequence_images_path)
        # read all dicom files for selected sequence
        dcm_files_found = glob.glob("*.dcm")
        if not dcm_files_found:
            dcm_files_found = glob.glob("*.IMA") # if sequence is IMA format instead of dcm
        # slice to be processed from selected sequence
        for slice in range(1, slices+1):
            current_slice = sequence + '_slice_' + str(slice)
            # single slice processing for DCE sequence (here selected slice number is 5)
            if current_slice not in [sequence + '_slice_5']:
                continue
            # read slice path to be processed
            slice_path = DATA_PATH + '/' + sequence_images_path + '/' + current_slice
            data = Path(slice_path)
            # list of all DICOMs to be processed for the selected slice (example: slice number = 5 here)
            lstFilesDCM = list(data.glob('**/*.IMA')) 
    
            # read all dicom files for the selected sequence and slice
            files, ArrayDicomiBEAt, filenameDCM = read_DICOM_files(lstFilesDCM)
            # get sitk image parameters for registration (pixel spacing)
            image_parameters = get_sitk_image_details_from_DICOM(slice_path)
            # sort slices correctly - based on acquisition time for model driven registration
            sorted_slice_files = sort_all_slice_files_acquisition_time(files)
            # run DCE MDR test 
            iBEAt_test_DCE(Elastix_Parameter_file_PATH, output_dir, sorted_slice_files, ArrayDicomiBEAt, image_parameters, AIFs_PATH, patient_folder)


# test DCE using model driven registration
def iBEAt_test_DCE(Elastix_Parameter_file_PATH, output_dir, sorted_slice_files, ArrayDicomiBEAt, image_parameters, AIFs_PATH, patient_folder):
    """ Example application of MDR in renal DCE (iBEAt data)  

    Description
    -----------
    This function performs model driven registration for selected DCE sequence on a single selected slice 
    and returns as output the MDR registered images, signal model fit, deformation field x, deformation field y, 
    fitted parameters Fp, Tp, Ps, Te, and the final diagnostics.
    
    Args
    ----
    Elastix_Parameter_file_PATH (string): complete path to the Elastix parameter file to be used.  
    output_dir (string): directory where results are saved.  
    slice_sorted_files (list): selected slices to process using MDR - sorted according to acquisition time.   
    ArrayDicomiBEAt (numpy.ndarray): input DICOM to numpy array (unsorted).  
    image_parameters (stik tuple): distance between pixels (in mm) along each dimension.  
    AIFs_PATH (string): string with full AIF path.  
    patient_folder (string): patient folder with AIFs text file.  
    """
    
    start_computation_time = time.time()
    # define numpy array with same input shape as original DICOMs
    image_shape = np.shape(ArrayDicomiBEAt)
    original_images = np.zeros(image_shape)

    # initialise original_images with sorted acquisiton times to run MDR
    for i, s in enumerate(sorted_slice_files):
        img2d = s.pixel_array
        original_images[:, :, i] = img2d

    # read module file for the DCE model as string
    full_module_name = "models.two_compartment_filtration_model_DCE"
    # generate a module named as a string
    model = importlib.import_module(full_module_name)
    # read signal model parameters
    signal_model_parameters = read_signal_model_parameters(AIFs_PATH, patient_folder)
    # read elastix model parameters
    elastix_model_parameters = read_elastix_model_parameters(Elastix_Parameter_file_PATH, ['MaximumNumberOfIterations', 256])
    
    #Perform MDR
    MDR_output = model_driven_registration(original_images, image_parameters, model, signal_model_parameters, elastix_model_parameters, precision = 30, function = 'main', log = False)

    #Export results
    export_images(MDR_output[0], output_dir +'/coregistered/MDR-registered_DCE_')
    export_images(MDR_output[1], output_dir +'/fit/fit_image_')
    export_images(MDR_output[2][:,:,0,:], output_dir +'/deformation_field/final_deformation_x_')
    export_images(MDR_output[2][:,:,1,:], output_dir +'/deformation_field/final_deformation_y_')
    export_maps(MDR_output[3][0,:], output_dir + '/fitted_parameters/Fp', np.shape(original_images))
    export_maps(MDR_output[3][1,:], output_dir + '/fitted_parameters/Tp', np.shape(original_images))
    export_maps(MDR_output[3][2,:], output_dir + '/fitted_parameters/Ps', np.shape(original_images))
    export_maps(MDR_output[3][3,:], output_dir + '/fitted_parameters/Te', np.shape(original_images))
    MDR_output[4].to_csv(output_dir + 'DCE_largest_deformations.csv')

    # Report computation times
    end_computation_time = time.time()
    print("total computation time for MDR (minutes taken:)...")
    print(0.0166667*(end_computation_time - start_computation_time)) # in minutes
    print("completed MDR registration!")
    print("Finished processing Model Driven Registration case for iBEAt study DCE sequence!")


## read sequence acquisition parameter for signal modelling
def read_signal_model_parameters(AIFs_PATH, patient_folder):
   
    aif, times = load_txt(AIFs_PATH + '/' + str(patient_folder) + '/' + 'AIF__2C Filtration__Curve.txt')
    aif.append(aif[-1])
    times.append(times[-1])
    # user defined parameters
    timepoint = 15
    Hct = 0.45
    # input signal model parameters
    signal_model_parameters = [aif, times, timepoint, Hct]
    return signal_model_parameters


def load_txt(full_path_txt):
    """ reads the AIF text file to find the AIF values and associated time-points.  

        Args
        ----
        full_path_txt (string): file path to the AIF text file.  

        Returns
        -------
        aif (list): arterial input function at each timepoint.  
        time (list): corresponding timepoints at each AIF.    
    """
    counter_file = open(full_path_txt, 'r+')
    content_lines = []
    for cnt, line in enumerate(counter_file):
        content_lines.append(line)
    x_values_index = content_lines.index('X-values\n')
    assert (content_lines[x_values_index+1]=='\n')
    y_values_index = content_lines.index('Y-values\n')
    assert (content_lines[y_values_index+1]=='\n')
    time = list(map(lambda x: float(x), content_lines[x_values_index+2 : y_values_index-1]))
    aif = list(map(lambda x: float(x), content_lines[y_values_index+2 :]))
    return aif, time