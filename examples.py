"""
Example uses of mdreg
"""

import os
import numpy as np
import pandas as pd

from dbdicom import Folder
from mdreg import MDReg
import mdreg.models as mdl

data = os.path.join(os.path.dirname(__file__), 'data')
results = os.path.join(os.path.dirname(__file__), 'results')
elastix_pars = os.path.join(os.path.dirname(__file__), 'elastix')

# To read and write from/to other locations, set these path names
data = 'C:\\Users\\steve\\Dropbox\\Data\\mdr_data'
results = 'C:\\Users\\steve\\Dropbox\\Data\\mdr_results'



def fit_DTI():

    print('Reading data..')
    folder = Folder(data).open()
    sortby = ['SliceLocation', 'AcquisitionTime']
    array, header = folder.series(1).array(sortby, pixels_first=True)
    slice = 15
    b_values = [float(hdr[(0x19, 0x100c)]) for hdr in header[slice,:,0]]
    b_vectors = [hdr[(0x19, 0x100e)] for hdr in header[slice,:,0]]
    orientation = [hdr.ImageOrientationPatient for hdr in header[slice,:,0]] 
    
    mdr = MDReg()
    mdr.set_array(np.squeeze(array[:,:,slice,:,0]))
    mdr.signal_parameters = [b_values, b_vectors, orientation]
    mdr.signal_model = mdl.DTI
    mdr.pixel_spacing = header[slice,0,0].PixelSpacing
    mdr.read_elastix(os.path.join(elastix_pars, 'BSplines_DTI.txt'))
    mdr.set_elastix(MaximumNumberOfIterations = 256)
    mdr.precision = 1
    mdr.export_path = os.path.join(results, 'DTI')
    mdr.fit()
    mdr.export()

    folder.close()


def fit_DWI_simple():
 
    print('Reading data..')
    folder = Folder(data).open()
    sortby = ['SliceLocation', 'AcquisitionTime']
    array, header = folder.series(2).array(sortby, pixels_first=True)
    slice = 15

    mdr = MDReg()
    mdr.set_array(np.squeeze(array[:,:,slice,:,0]))
    mdr.signal_parameters = [0,10.000086, 19.99908294, 30.00085926, 50.00168544, 80.007135, 100.0008375, 199.9998135, 300.0027313, 600.0]
    mdr.pixel_spacing = header[slice,0,0].PixelSpacing
    mdr.signal_model = mdl.DWI_simple
    mdr.read_elastix(os.path.join(elastix_pars, 'BSplines_IVIM.txt'))
    mdr.set_elastix(MaximumNumberOfIterations = 256)
    mdr.precision = 1
    mdr.export_path = os.path.join(results, 'DWI_simple')
    mdr.fit()
    mdr.export()

    folder.close()


def fit_DWI():
 
    print('Reading data..')
    folder = Folder(data).open()
    sortby = ['SliceLocation', 'AcquisitionTime']
    array, header = folder.series(2).array(sortby, pixels_first=True)
    slice = 15

    mdr = MDReg()
    mdr.set_array(np.squeeze(array[:,:,slice,:,0]))
    mdr.signal_parameters = [0,10.000086, 19.99908294, 30.00085926, 50.00168544, 80.007135, 100.0008375, 199.9998135, 300.0027313, 600.0]
    mdr.pixel_spacing = header[slice,0,0].PixelSpacing
    mdr.signal_model = mdl.DWI
    mdr.read_elastix(os.path.join(elastix_pars, 'BSplines_IVIM.txt'))
    mdr.set_elastix(MaximumNumberOfIterations = 256)
    mdr.precision = 1
    mdr.export_path = os.path.join(results, 'DWI')
    mdr.fit()
    mdr.export()

    folder.close()


def fit_T1_simple():

    print('Reading data..')
    folder = Folder(data).open()
    sortby = ['SliceLocation', 'InversionTime']
    array, header = folder.series(3).array(sortby, pixels_first=True)
    slice = 2

    mdr = MDReg()
    mdr.set_array(np.squeeze(array[:,:,slice,:,0]))
    mdr.signal_parameters = [hdr.InversionTime for hdr in header[slice,:,0]]
    mdr.pixel_spacing = header[slice,0,0].PixelSpacing
    mdr.signal_model = mdl.T1_simple
    mdr.read_elastix(os.path.join(elastix_pars, 'BSplines_T1.txt'))
    mdr.set_elastix(MaximumNumberOfIterations = 256)
    mdr.precision = 1
    mdr.export_path = os.path.join(results, 'T1_simple')
    mdr.fit()
    mdr.export()

    folder.close()


def fit_T1():

    print('Reading data..')
    folder = Folder(data).open()
    sortby = ['SliceLocation', 'InversionTime']
    array, header = folder.series(3).array(sortby, pixels_first=True)
    slice = 2

    mdr = MDReg()
    mdr.set_array(np.squeeze(array[:,:,slice,:,0]))
    mdr.signal_parameters = [hdr.InversionTime for hdr in header[slice,:,0]]
    mdr.pixel_spacing = header[slice,0,0].PixelSpacing
    mdr.signal_model = mdl.T1
    mdr.read_elastix(os.path.join(elastix_pars, 'BSplines_T1.txt'))
    mdr.set_elastix(MaximumNumberOfIterations = 256)
    mdr.precision = 1
    mdr.export_path = os.path.join(results, 'T1')
    mdr.fit()
    mdr.export()

    folder.close()


def fit_constant():

    print('Reading data..')
    folder = Folder(data).open()
    sortby = ['SliceLocation', 'AcquisitionTime']
    array, header = folder.series(3).array(sortby, pixels_first=True)
    slice = 2

    mdr = MDReg()
    mdr.set_array(np.squeeze(array[:,:,slice,:,0]))
    mdr.signal_parameters = [0,30,40,50,60,70,80,90,100,110,120]
    mdr.pixel_spacing = header[slice,0,0].PixelSpacing
    mdr.signal_model = mdl.constant
    mdr.read_elastix(os.path.join(elastix_pars, 'BSplines_MT.txt'))
    mdr.set_elastix(MaximumNumberOfIterations = 256)
    mdr.precision = 10
    mdr.export_path = os.path.join(results, 'constant')
    mdr.fit()
    mdr.export()

    folder.close()


def fit_T2_simple():

    print('Reading data..')
    folder = Folder(data).open()
    sortby = ['SliceLocation', 'AcquisitionTime']
    array, header = folder.series(4).array(sortby, pixels_first=True)
    slice = 2

    mdr = MDReg()
    mdr.set_array(np.squeeze(array[:,:,slice,:,0]))
    mdr.signal_parameters = [0,30,40,50,60,70,80,90,100,110,120]
    mdr.pixel_spacing = header[slice,0,0].PixelSpacing
    mdr.signal_model = mdl.T2_simple
    mdr.read_elastix(os.path.join(elastix_pars, 'BSplines_T2_simple.txt'))
    mdr.set_elastix(MaximumNumberOfIterations = 256)
    mdr.precision = 1
    mdr.export_path = os.path.join(results, 'T2_simple')
    mdr.fit()
    mdr.export()

    folder.close()


def fit_T2():

    print('Reading data..')
    folder = Folder(data).open()
    sortby = ['SliceLocation', 'AcquisitionTime']
    array, header = folder.series(4).array(sortby, pixels_first=True)
    slice = 2

    mdr = MDReg()
    mdr.set_array(np.squeeze(array[:,:,slice,:,0]))
    mdr.signal_parameters = [0,30,40,50,60,70,80,90,100,110,120]
    mdr.pixel_spacing = header[slice,0,0].PixelSpacing
    mdr.signal_model = mdl.T2
    mdr.read_elastix(os.path.join(elastix_pars, 'BSplines_T2.txt'))
    mdr.set_elastix(MaximumNumberOfIterations = 256)
    mdr.precision = 1
    mdr.export_path = os.path.join(results, 'T2')
    mdr.fit()
    mdr.export()

    folder.close()


def fit_T2star_simple():

    print('Reading data..') 
    folder = Folder(data).open()
    sortby = ['SliceLocation', 'AcquisitionTime']
    array, header = folder.series(5).array(sortby, pixels_first=True)
    slice = 2

    mdr = MDReg()
    mdr.set_array(np.squeeze(array[:,:,slice,:,0]))
    mdr.signal_parameters = [hdr.EchoTime for hdr in header[slice,:,0]]
    mdr.pixel_spacing = header[slice,0,0].PixelSpacing
    mdr.signal_model = mdl.T2star_simple
    mdr.read_elastix(os.path.join(elastix_pars, 'BSplines_T2star.txt'))
    mdr.set_elastix(MaximumNumberOfIterations = 256)
    mdr.precision = 10
    mdr.export_path = os.path.join(results, 'T2star_simple')
    mdr.fit()
    mdr.export()

    folder.close()


def fit_T2star():

    print('Reading data..')
    folder = Folder(data).open()
    sortby = ['SliceLocation', 'AcquisitionTime']
    array, header = folder.series(5).array(sortby, pixels_first=True)
    slice = 2

    mdr = MDReg()
    mdr.set_array(np.squeeze(array[:,:,slice,:,0]))
    mdr.signal_parameters = [hdr.EchoTime for hdr in header[slice,:,0]]
    mdr.pixel_spacing = header[slice,0,0].PixelSpacing
    mdr.signal_model = mdl.T2star
#    mdr.read_elastix(os.path.join(elastix_pars, 'BSplines_T2star.txt'))
#    mdr.set_elastix(MaximumNumberOfIterations = 256)
    mdr.precision = 1
    mdr.export_path = os.path.join(results, 'T2star')
    mdr.fit()
    mdr.export()

    folder.close()


def fit_T2star_model():
    """Fit the data to a signal model without motion correction"""

    print('Reading data..')
    folder = Folder(data).open()
    sortby = ['SliceLocation', 'AcquisitionTime']
    array, header = folder.series(5).array(sortby, pixels_first=True)
    slice = 2

    mdr = MDReg()
    mdr.set_array(np.squeeze(array[:,:,slice,:,0]))
    mdr.signal_parameters = [hdr.EchoTime for hdr in header[slice,:,0]]
    mdr.signal_model = mdl.T2star
    mdr.fit_signal()
    mdr.export_path = os.path.join(results, 'T2starmodel')
    mdr.export_data()
    mdr.export_fit()

    folder.close()


def fit_DCE_2CFM_model():

    print('Reading data..')
    folder = Folder(data).open()
    sortby = ['SliceLocation', 'AcquisitionTime']
    array, _ = folder.series(0).array(sortby, pixels_first=True)
    file = os.path.join(data, 'AIF.csv')
    aif = pd.read_csv(file).to_numpy()
    baseline = 15
    hematocrit = 0.45
    slice = 4

    mdr = MDReg()
    mdr.set_array(np.squeeze(array[:,:,slice,:,0]))
    mdr.signal_parameters = [np.squeeze(aif[:,1]), np.squeeze(aif[:,0]), baseline, hematocrit]
    mdr.signal_model = mdl.DCE_2CFM
    mdr.fit_signal()
    mdr.export_path = os.path.join(results, 'DCEmodel')
    mdr.export_data()
    mdr.export_fit()

    folder.close()



if __name__ == '__main__':

#    fit_T2star_model()
#    fit_DCE_2CFM_model()

#    fit_DWI()
#    fit_DWI_simple()
#    fit_T1_simple()
#    fit_T1()
#    fit_T2_simple()
#    fit_T2()
    fit_T2star_simple()
#    fit_T2star()
#    fit_DCE_2CFM()
#    fit_DTI()
#    fit_constant()   
    