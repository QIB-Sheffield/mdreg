"""
Example use of 3D mdreg
"""

import os
import numpy as np

from dbdicom.folder import Folder
from mdreg import MDReg
from mdreg import models as mdl

data = os.path.join(os.path.dirname(__file__), 'data')
results = os.path.join(os.path.dirname(__file__), 'results')
elastix_pars = os.path.join(os.path.dirname(__file__), 'elastix')

# To read and write from/to other locations, set these path names
## 3D datasets
data = 'C:\\Users\\md1ksha\\Documents\\GitHub\data\\3D\\Kanishka'
results = 'C:\\Users\\md1ksha\\Documents\\GitHub\\dev\\results_3D_MDREG'

def fit_constant():
    
    print('Reading data..')
    folder = Folder(data).open()
    sortby = ['SliceLocation', 'AcquisitionTime']
    array, header = folder.series(0).array(sortby, pixels_first=True)
    mdr = MDReg()
    mdr.set_array(np.squeeze(array[:,:,:,:,0])) # 3D # (78, 96, 36, 28, 1)

    mdr.pixel_spacing = header[0,0,0].PixelSpacing # 3D volume spacing (x,y)
    mdr.slice_thickness = header[0,0,0].SliceThickness # 3D volume spacing (x,y,z) 
    mdr.pixel_spacing.append(mdr.slice_thickness) # required for 3D pixel spacing

    mdr.signal_model = mdl.constant
    mdr.set_elastix(MaximumNumberOfIterations = 256)
    mdr.precision = 10
    mdr.export_path = os.path.join(results, 'constant')
    mdr.fit()
    mdr.export()

    folder.close()


if __name__ == '__main__':

    fit_constant()   
    