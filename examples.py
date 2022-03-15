"""
MDR tutorial showing how to motion-correct DICOM T2* data.

The tutorial uses the DICOM example data provided *here*. 

The `dicomdb` package is used to read the DICOM data 
and return numpy arrays. 

In order to run the tutorial as is, extract the data in a folder 
"mydata" in the same directory as this script.
The results will be saved in a folder "myresults".

In order to read and write from other locations, 
change the path names in the script below.

Then just run this module as a script.
"""

import os, time
import numpy as np
import pandas as pd

from dbdicom import Folder

from MDR.MDR import model_driven_registration
from MDR.Tools import read_elastix_model_parameters, export_results

import models_signal.two_compartment_filtration_model_DCE as DCE
import models_signal.constant as constant
import models_signal.DTI as DTI
import models_signal.DWI_monoexponential as DWI_monoexponential
import models_signal.DWI_monoexponential_simple as DWI_monoexponential_simple
import models_signal.T1 as T1
import models_signal.T1_simple as T1_simple
import models_signal.T2 as T2
import models_signal.T2_simple as T2_simple
import models_signal.T2star as T2star
import models_signal.T2star_simple as T2star_simple

# To read and write from/to other locations, change these path names
data = os.path.join(os.path.dirname(__file__), 'data')
results = os.path.join(os.path.dirname(__file__), 'results')


def fit_DCE():

    coreg_model = 'BSplines_DCE' 
    signal_model = DCE
    model_name = 'DCE_2CFM' 
    par_name = ['FP', 'TP', 'PS', 'TE'] 
    series_nr = 0 
    slice = 4

    print('Reading data..')
    folder = Folder(data).open()
    sortby = ['SliceLocation', 'AcquisitionTime']
    array, header = folder.series(series_nr).array(sortby, pixels_first=True)

    file = os.path.join(data, 'test_data', 'AIF.csv')
    array = pd.read_csv(file).to_numpy()
    baseline = 15
    Hct = 0.45
    signal_model_parameters = [array[:,1], array[:,0], baseline, Hct]

    print('Performing MDR..')
    start = time.time()
    file = os.path.join(os.getcwd(), 'models_coreg', coreg_model + '.txt')
    MDR_output = model_driven_registration(
        np.squeeze(array[:,:,slice,:,0]),                       # images
        header[slice,0,0].PixelSpacing,                         # pixel size
        signal_model,                                           # model
        signal_model_parameters,             # signal model parameters
        read_elastix_model_parameters(file, ['MaximumNumberOfIterations', 256]), 
        precision = 1, 
    )
    print('Calculation time (min)', (time.time()-start)/60)

    print('Exporting results..')
    export_results(
        MDR_output = MDR_output, 
        path = results, 
        model = model_name, 
        pars = par_name, 
        xy = np.shape(np.squeeze(array[:,:,slice,:,0])), 
    )
    folder.close()


def fit_DTI():

    coreg_model = 'BSplines_DTI' 
    signal_model = DTI 
    model_name = 'DTI' 
    par_name = ['FA', 'ADC'] 
    series_nr = 1
    slice = 15

    print('Reading data..')
    folder = Folder(data).open()
    sortby = ['SliceLocation', 'AcquisitionTime']
    array, header = folder.series(series_nr).array(sortby, pixels_first=True)

    b_values = [hdr[(0x19, 0x100c)] for hdr in header[slice,:,0]]
    b_Vec_original = [hdr[(0x19, 0x100e)] for hdr in header[slice,:,0]]
    image_orientation_patient = [hdr.ImageOrientationPatient for hdr in header[slice,:,0]]
    signal_model_parameters = [b_values, b_Vec_original, image_orientation_patient]

    print('Performing MDR..')
    start = time.time()
    file = os.path.join(os.getcwd(), 'models_coreg', coreg_model + '.txt')
    MDR_output = model_driven_registration(
        np.squeeze(array[:,:,slice,:,0]),                       # images
        header[slice,0,0].PixelSpacing,                         # pixel size
        signal_model,                                           # model
        signal_model_parameters,    
        read_elastix_model_parameters(file, ['MaximumNumberOfIterations', 1024]), 
        precision = 1, 
    )
    print('Calculation time (min)', (time.time()-start)/60)

    print('Exporting results..')
    export_results(
        MDR_output = MDR_output, 
        path = results, 
        model = model_name, 
        pars = par_name, 
        xy = np.shape(np.squeeze(array[:,:,slice,:,0])), 
    )
    folder.close()


def fit_DWI_monoexponential_simple():

    coreg_model = 'BSplines_IVIM' 
    signal_model = DWI_monoexponential_simple 
    model_name = 'DWI_simple' 
    par_name = ['S0', 'ADC'] 
    series_nr = 2
    slice = 15

    print('Reading data..')
    folder = Folder(data).open()
    folder.print()

    sortby = ['SliceLocation', 'AcquisitionTime']
    array, header = folder.series(series_nr).array(sortby, pixels_first=True)
    bvalues = [0,10.000086, 19.99908294, 30.00085926, 50.00168544, 80.007135, 100.0008375, 199.9998135, 300.0027313, 600.0]
    signal_model_parameters = bvalues

    print('Performing MDR..')
    start = time.time()
    file = os.path.join(os.getcwd(), 'models_coreg', coreg_model + '.txt')
    MDR_output = model_driven_registration(
        np.squeeze(array[:,:,slice,:,0]),                       # images
        header[slice,0,0].PixelSpacing,                         # pixel size
        signal_model,                                           # model
        signal_model_parameters,    
        read_elastix_model_parameters(file, ['MaximumNumberOfIterations', 256]), 
        precision = 1, 
    )
    print('Calculation time (min)', (time.time()-start)/60)

    print('Exporting results..')
    export_results(
        MDR_output = MDR_output, 
        path = results, 
        model = model_name, 
        pars = par_name, 
        xy = np.shape(np.squeeze(array[:,:,slice,:,0])), 
    )
    folder.close()


def fit_DWI_monoexponential():

    coreg_model = 'BSplines_IVIM' 
    signal_model = DWI_monoexponential
    model_name = 'DWI' 
    par_name = ['S0', 'ADC'] 
    series_nr = 2
    slice = 15

    print('Reading data..')
    folder = Folder(data).open()
    sortby = ['SliceLocation', 'AcquisitionTime']
    array, header = folder.series(series_nr).array(sortby, pixels_first=True)
    bvalues = [0,10.000086, 19.99908294, 30.00085926, 50.00168544, 80.007135, 100.0008375, 199.9998135, 300.0027313, 600.0]
    signal_model_parameters = bvalues + bvalues + bvalues

    print('Performing MDR..')
    start = time.time()
    file = os.path.join(os.getcwd(), 'models_coreg', coreg_model + '.txt')
    MDR_output = model_driven_registration(
        np.squeeze(array[:,:,slice,:,0]),                       # images
        header[slice,0,0].PixelSpacing,                         # pixel size
        signal_model,                                           # model
        signal_model_parameters,    
        read_elastix_model_parameters(file, ['MaximumNumberOfIterations', 256]), 
        precision = 1, 
    )
    print('Calculation time (min)', (time.time()-start)/60)

    print('Exporting results..')
    export_results(
        MDR_output = MDR_output, 
        path = results, 
        model = model_name, 
        pars = par_name, 
        xy = np.shape(np.squeeze(array[:,:,slice,:,0])), 
    )
    folder.close()


def fit_T1_simple():

    coreg_model = 'BSplines_T1' 
    signal_model = T1_simple 
    model_name = 'T1_simple' 
    par_name = ['S0', 'inversion_efficiency', 'T1'] 
    series_nr = 3
    slice = 2

    print('Reading data..')
    folder = Folder(data).open()
    sortby = ['SliceLocation', 'InversionTime']
    array, header = folder.series(series_nr).array(sortby, pixels_first=True)
    signal_model_parameters = [hdr.InversionTime for hdr in header[slice,:,0]]

    print('Performing MDR..')
    start = time.time()
    file = os.path.join(os.getcwd(), 'models_coreg', coreg_model + '.txt')
    MDR_output = model_driven_registration(
        np.squeeze(array[:,:,slice,:,0]),                       # images
        header[slice,0,0].PixelSpacing,                         # pixel size
        signal_model,                                           # model
        signal_model_parameters,                                # T2 prep times (ms)
        read_elastix_model_parameters(file, ['MaximumNumberOfIterations', 256]), 
        precision = 1, 
    )
    print('Calculation time (min)', (time.time()-start)/60)

    print('Exporting results..')
    export_results(
        MDR_output = MDR_output, 
        path = results, 
        model = model_name, 
        pars = par_name, 
        xy = np.shape(np.squeeze(array[:,:,slice,:,0])), 
    )
    folder.close()


def fit_T1():

    coreg_model = 'BSplines_T1' 
    signal_model = T1 
    model_name = 'T1' 
    par_name = ['T1', 'T1app', 'B', 'A'] 
    series_nr = 3 
    slice = 2

    print('Reading data..')
    folder = Folder(data).open()
    sortby = ['SliceLocation', 'InversionTime']
    array, header = folder.series(series_nr).array(sortby, pixels_first=True)
    signal_model_parameters = [hdr.InversionTime for hdr in header[slice,:,0]]
   # export_animation(np.squeeze(array[:,:,slice,:,0]), results, 'T1data')

    print('Performing MDR..')
    start = time.time()
    file = os.path.join(os.getcwd(), 'models_coreg', coreg_model + '.txt')
    MDR_output = model_driven_registration(
        np.squeeze(array[:,:,slice,:,0]),                       # images
        header[slice,0,0].PixelSpacing,                         # pixel size
        signal_model,                                           # model
        signal_model_parameters,                                # T2 prep times (ms)
        read_elastix_model_parameters(file, ['MaximumNumberOfIterations', 256]), 
        precision = 1, 
    )
    print('Calculation time (min)', (time.time()-start)/60)

    print('Exporting results..')
    export_results(
        MDR_output = MDR_output, 
        path = results, 
        model = model_name, 
        pars = par_name, 
        xy = np.shape(np.squeeze(array[:,:,slice,:,0])), 
    )
    folder.close()


def fit_constant():

    coreg_model = 'BSplines_MT' 
    signal_model = constant 
    model_name = 'constant' 
    par_name = ['mean'] 
    series_nr = 3
    slice = 2

    print('Reading data..')
    folder = Folder(data).open()
    sortby = ['SliceLocation', 'AcquisitionTime']
    array, header = folder.series(series_nr).array(sortby, pixels_first=True)

    print('Performing MDR..')
    start = time.time()
    file = os.path.join(os.getcwd(), 'models_coreg', coreg_model + '.txt')
    MDR_output = model_driven_registration(
        np.squeeze(array[:,:,slice,:,0]),                       # images
        header[slice,0,0].PixelSpacing,                         # pixel size
        signal_model,                                           # model
        None,                                 
        read_elastix_model_parameters(file, ['MaximumNumberOfIterations', 256]), 
        precision = 1, 
    )
    print('Calculation time (min)', (time.time()-start)/60)

    print('Exporting results..')
    export_results(
        MDR_output = MDR_output, 
        path = results, 
        model = model_name, 
        pars = par_name, 
        xy = np.shape(np.squeeze(array[:,:,slice,:,0])), 
    )
    folder.close()


def fit_T2_simple():

    coreg_model = 'BSplines_T2' 
    signal_model = T2_simple 
    model_name = 'T2_simple' 
    par_name = ['S0', 'T2'] 
    series_nr = 4 
    slice = 2

    print('Reading data..')
    folder = Folder(data).open()
    sortby = ['SliceLocation', 'AcquisitionTime']
    array, header = folder.series(series_nr).array(sortby, pixels_first=True)
    signal_model_parameters = [0,30,40,50,60,70,80,90,100,110,120]

    print('Performing MDR..')
    start = time.time()
    file = os.path.join(os.getcwd(), 'models_coreg', coreg_model + '.txt')
    MDR_output = model_driven_registration(
        np.squeeze(array[:,:,slice,:,0]),                       # images
        header[slice,0,0].PixelSpacing,                         # pixel size
        signal_model,                                           # model
        signal_model_parameters,                                # T2 prep times (ms)
        read_elastix_model_parameters(file, ['MaximumNumberOfIterations', 256]), 
        precision = 1, 
    )
    print('Calculation time (min)', (time.time()-start)/60)

    print('Exporting results..')
    export_results(
        MDR_output = MDR_output, 
        path = results, 
        model = model_name, 
        pars = par_name, 
        xy = np.shape(np.squeeze(array[:,:,slice,:,0])), 
    )
    folder.close()


def fit_T2():

    coreg_model = 'BSplines_T2' 
    signal_model = T2
    model_name = 'T2' 
    par_name = ['S0', 'T2'] 
    series_nr = 4 
    slice = 2

    print('Reading data..')
    folder = Folder(data).open()
    sortby = ['SliceLocation', 'AcquisitionTime']
    array, header = folder.series(series_nr).array(sortby, pixels_first=True)
    signal_model_parameters = [0,30,40,50,60,70,80,90,100,110,120]

    print('Performing MDR..')
    start = time.time()
    file = os.path.join(os.getcwd(), 'models_coreg', coreg_model + '.txt')
    MDR_output = model_driven_registration(
        np.squeeze(array[:,:,slice,:,0]),                       # images
        header[slice,0,0].PixelSpacing,                         # pixel size
        signal_model,                                           # model
        signal_model_parameters,                   # T2 prep times (ms)
        read_elastix_model_parameters(file, ['MaximumNumberOfIterations', 256]), 
        precision = 1, 
    )
    print('Calculation time (min)', (time.time()-start)/60)

    print('Exporting results..')
    export_results(
        MDR_output = MDR_output, 
        path = results, 
        model = model_name, 
        pars = par_name, 
        xy = np.shape(np.squeeze(array[:,:,slice,:,0])), 
    )
    folder.close()


def fit_T2star_simple():

    coreg_model = 'BSplines_T2star' 
    signal_model = T2star_simple 
    model_name = 'T2_star_simple' 
    par_name = ['S0', 'T2star'] 
    series_nr = 5
    slice = 2

    print('Reading data..')
    folder = Folder(data).open()
    sortby = ['SliceLocation', 'AcquisitionTime']
    array, header = folder.series(series_nr).array(sortby, pixels_first=True)
    signal_model_parameters = [hdr.EchoTime for hdr in header[slice,:,0]]

    print('Performing MDR..')
    start = time.time()
    file = os.path.join(os.getcwd(), 'models_coreg', coreg_model + '.txt')
    MDR_output = model_driven_registration(
        np.squeeze(array[:,:,slice,:,0]),                       # images
        header[slice,0,0].PixelSpacing,                         # pixel size
        signal_model,                                           # model
        signal_model_parameters,             # signal model parameters
        read_elastix_model_parameters(file, ['MaximumNumberOfIterations', 256]), 
        precision = 1, 
    )
    print('Calculation time (min)', (time.time()-start)/60)

    print('Exporting results..')
    export_results(
        MDR_output = MDR_output, 
        path = results, 
        model = model_name, 
        pars = par_name, 
        xy = np.shape(np.squeeze(array[:,:,slice,:,0])), 
    )
    folder.close()


def fit_T2star():

    coreg_model = 'BSplines_T2star' 
    signal_model = T2star
    model_name = 'T2_star' 
    par_name = ['S0', 'T2star'] 
    series_nr = 5 
    slice = 2

    print('Reading data..')
    folder = Folder(data).open()
    sortby = ['SliceLocation', 'AcquisitionTime']
    array, header = folder.series(series_nr).array(sortby, pixels_first=True)
    signal_model_parameters = [hdr.EchoTime for hdr in header[slice,:,0]]

    print('Performing MDR..')
    start = time.time()
    file = os.path.join(os.getcwd(), 'models_coreg', coreg_model + '.txt')
    MDR_output = model_driven_registration(
        np.squeeze(array[:,:,slice,:,0]),                       # images
        header[slice,0,0].PixelSpacing,                         # pixel size
        signal_model,                                           # model
        signal_model_parameters,             # signal model parameters
        read_elastix_model_parameters(file, ['MaximumNumberOfIterations', 256]), 
        precision = 1, 
    )
    print('Calculation time (min)', (time.time()-start)/60)

    print('Exporting results..')
    export_results(
        MDR_output = MDR_output, 
        path = results, 
        model = model_name, 
        pars = par_name, 
        xy = np.shape(np.squeeze(array[:,:,slice,:,0])), 
    )
    folder.close()


if __name__ == '__main__':

    # fit_DCE()
    # fit_constant()
    # fit_DTI()
    # fit_DWI_monoexponential()
    # fit_DWI_monoexponential_simple()
    fit_T1_simple()
    # fit_T1()
    # fit_T2_simple()
    # fit_T2()
    # fit_T2star_simple()
    # fit_T2star()
    