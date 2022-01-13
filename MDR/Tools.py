"""
TOOLS FOR MODEL DRIVEN REGISTRATION (MDR)
This file contains several functions that help and support the development of a motion correction analysis pipeline

MDR Library
@Kanishka Sharma
@Joao Almeida e Sousa
@Steven Sourbron
2021
"""

import os
import numpy as np
import importlib
from PIL import Image
import pydicom
import SimpleITK as sitk
import itk


def read_DICOM_files(lstFilesDCM):
    """ Reads input DICOM Files.

    Args:
    ----
        lstFilesDCM (list): List containing the file paths of the DICOM Files.

    Returns:
    -------
        files (list): List containing the pydicom datasets.
        ArrayDicom (numpy.array): Image resulting from the stack of the DICOM files in lstFilesDCM.
        filenameDCM (string): Last element of lstFilesDCM. Should be a file path to a DICOM file.
    """
    files = []
    RefDs = pydicom.dcmread(lstFilesDCM[0])
    SeriesPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))           
    ArrayDicom = np.zeros(SeriesPixelDims, dtype=RefDs.pixel_array.dtype)

    # read all dicoms and output dicom files
    for filenameDCM in lstFilesDCM: 
        files.append(pydicom.dcmread(filenameDCM))
        ds = pydicom.dcmread(filenameDCM)
        # write pixel data into numpy array
        ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array  

    return files, ArrayDicom, filenameDCM


def get_sitk_image_details_from_DICOM(filenameDCM):
    """ Reads and returns image spacing of the input DICOM File.

    Args:
    ----
        filenameDCM (string): File path to a DICOM File.

    Returns:
    -------
        spacing (float): Float value describing the space between pixels in the given image.
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(filenameDCM)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    spacing = image.GetSpacing() 
    return spacing


def sort_all_slice_files_acquisition_time(files):
    """ Sort the DICOM files based on acquisition time.

    Args:
    ----
        files (list): List containing the pydicom datasets of the DICOM Files.

    Returns:
    -------
        slice_sorted_acq_time (list): List containing the file paths of the DICOM Files sorted by acquisition time.
    """
    slice_sorted_acq_time = []
    skipcount = 0
    for f in files: 
        if hasattr(f, 'AcquisitionTime'):
            slice_sorted_acq_time.append(f)
        else:
            skipcount = skipcount + 1
    print("skipped, no AcquisitionTime: {}".format(skipcount))

    return sorted(slice_sorted_acq_time, key=lambda s: s.AcquisitionTime) 


def read_elastix_model_parameters(Elastix_Parameter_file_PATH, *argv):
    """ Read elastix parameters from given text file.

    Args:
    ----
        Elastix_Parameter_file_PATH (string): Path to text file containing motion correction parameters.

    Returns:
    -------
        elastix_model_parameters (itk-elastix.ParameterObject): ITK-Elastix variable containing the parameters for the registration.

    Example:
    -------
        elastix_model_parameters = read_elastix_model_parameters('BSplines_T2star.txt', ['MaximumNumberOfIterations', 256], ['MaximumStepLength ', 1.0])

    """
    elastix_model_parameters = itk.ParameterObject.New()
    elastix_model_parameters.AddParameterFile(Elastix_Parameter_file_PATH)
    for lstParameters in argv:
        elastix_model_parameters.SetParameter(str(lstParameters[0]), str(lstParameters[1]))
    return elastix_model_parameters


def export_images(MDR_individual_output, folder):
    """ Save MDR results to given folder.

    Args:
    ----
        MDR_individual_output (numpy.array): Numpy array representing one of the outputs of MDR.
        folder (string): Path to the folder that will host the results/output of this method.
    """
    if not os.path.exists(os.path.dirname(folder)): os.makedirs(os.path.dirname(folder))
    shape = np.shape(MDR_individual_output)
    for i in range(shape[2]):
        im = Image.fromarray(MDR_individual_output[:,:,i])
        im.save(folder + str(i) + ".tiff")


def export_maps(MDR_individual_output, folder, shape):
    """ Save MDR results to given folder. Fitted Parameters to output folder.

    Args:
    ----
        MDR_individual_output (numpy.array): Numpy array representing one of the outputs of MDR.
        folder (string): Path to the folder that will host the results/output of this method.
        shape (list): Shape of the output array to which MDR_individual_output will be reshaped
    """
    if not os.path.exists(os.path.dirname(folder)): os.makedirs(os.path.dirname(folder))
    array = np.reshape(MDR_individual_output, [shape[0],shape[1]]) 
    Img = Image.fromarray(array)
    Img.save(folder + ".tiff")
