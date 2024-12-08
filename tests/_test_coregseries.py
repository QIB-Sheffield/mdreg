import os
import numpy as np
import mdreg
from mdreg import elastix, skimage


def test_skimage_series():

    print('Testing coreg series: elastix')

    data = mdreg.fetch('MOLLI')

    z = 0
    moving = data['array'][:,:,z,:]
    fixed = mdreg.fetch('MOLLIfit')['array'][:,:,z,:]

    coreg, defo = skimage.coreg_series(
        moving, fixed,
        params = {'attachment': 30},
    )
    mdreg.plot_series(
        moving, fixed, coreg, 
        vmin=0, vmax=10000, 
        show=True)


def test_elastix_series():

    print('Testing coreg series: elastix')

    data = mdreg.fetch('MOLLI')

    z = 0
    moving = data['array'][:,:,z,:]
    fixed = mdreg.fetch('MOLLIfit')['array'][:,:,z,:]

    params = elastix.params('bspline')
    params["FinalGridSpacingInPhysicalUnits"] = "5.0"

    coreg, defo = elastix.coreg_series(
        moving, fixed,
        params = params,
        spacing = data['pixel_spacing'],
    )
    mdreg.plot_series(
        moving, fixed, coreg, 
        path = os.path.join(os.getcwd(), 'tmp'), 
        filename='animation', 
        vmin=0, vmax=10000, 
        show=True)



if __name__ == '__main__':

#    test_elastix_series()
    test_skimage_series()

    print('------------------------------')
    print('coreg series testing: passed!!')
    print('------------------------------')