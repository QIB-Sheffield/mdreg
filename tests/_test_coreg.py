import numpy as np
import mdreg
from mdreg import elastix, skimage


def test_skimage():

    # Get data
    data = mdreg.fetch('MOLLI')
    fit = mdreg.fetch('MOLLIfit')

    # Coreg slice with fine grid spacing
    z, t = 0, 2
    coreg, defo = skimage.coreg(
        data['array'][:,:,z,t], 
        fit['array'][:,:,z,t],
        params = {'attachment': 30},
    )
    
    # Display result
    mdreg.plot_coreg(
        data['array'][:,:,z,t], 
        fit['array'][:,:,z,t], 
        coreg, defo)


def test_elastix():

    # Get data
    data = mdreg.fetch('MOLLI')
    fit = mdreg.fetch('MOLLIfit')

    # Coreg slice with fine grid spacing
    z, t = 0, 2
    coreg, defo = elastix.coreg(
        data['array'][:,:,z,t], 
        fit['array'][:,:,z,t],
        params = elastix.params(FinalGridSpacingInPhysicalUnits="5.0"),
        spacing = data['pixel_spacing'],
    )
    
    # Display result
    mdreg.plot_coreg(
        data['array'][:,:,z,t], 
        fit['array'][:,:,z,t], 
        coreg, defo)



if __name__ == '__main__':
        
#    test_elastix()
    test_skimage()