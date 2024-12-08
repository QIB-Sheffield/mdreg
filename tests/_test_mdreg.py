import os
import numpy as np
import mdreg


def my_model(xdata, S0, T1):
    return np.abs(S0 * (1 - 2 * np.exp(-xdata/T1)))

def my_model_init(xdata, signal, p0):
    S0 = np.amax(np.abs(signal))
    return [S0*p0[0], p0[1]]




def test_pixel_model():

    z = 0
    data = mdreg.fetch('MOLLI')
    moving = data['array'][:,:,z,:]
    vmax = 10000
    path = os.path.join(os.getcwd(), 'tmp')

    coreg, defo, fit, pars = mdreg.fit(moving,
        fit_pixel = {
            'model': my_model,
            'xdata': np.array(data['TI'])/1000,
            'func_init': my_model_init,
            'bounds': (0, np.inf),
            'p0': [1,1.3],
            'parallel': True,
#            'xtol': 1e-3,      
        },
        fit_coreg = {
            'spacing': data['pixel_spacing'],
            "FinalGridSpacingInPhysicalUnits": 5.0,
        },
        maxit=2, verbose=3,
        plot_params = {'path':path, 'vmin':0, 'vmax':vmax},
    )


def test_image_model():

    z = 0
    data = mdreg.fetch('MOLLI')
    moving = data['array'][:,:,z,:]
    vmax = 10000
    path = os.path.join(os.getcwd(), 'tmp')

    coreg, defo, fit, pars = mdreg.fit(moving,
        fit_image = {
            'func': mdreg.abs_exp_recovery_2p,
            'TI': np.array(data['TI'])/1000,
        },
        fit_coreg = {
            'spacing': data['pixel_spacing'],
            "FinalGridSpacingInPhysicalUnits": 5.0,
        },
        maxit=2, verbose=3,
        plot_params = {'path':path, 'vmin':0, 'vmax':vmax},
    )

def test_coreg_model():
    # getting started script

    data = mdreg.fetch('MOLLI')
    array = data['array'][:,:,0,:]
    molli = {
        'func': mdreg.abs_exp_recovery_2p,
        'TI': np.array(data['TI'])/1000,
        'progress_bar': True,
    }
    coreg, defo, fit, pars = mdreg.fit(array, fit_image=molli, verbose=2)
    mdreg.plot_series(array, fit, coreg, vmin=0, vmax=1e4, show=True)


def test_vfa_lin():

    data = mdreg.fetch(
        'VFA',
    )
    vfa_fit = {
        'func': mdreg.spgr_vfa_lin,     # VFA signal model
        'FA': data['FA'],               # Flip angle in degress 
        'progress_bar': True,   
    }
    coreg_params = {
        'spacing': data['spacing'],     # (x,y,z) voxel size in mm.
        'FinalGridSpacingInPhysicalUnits': 50.0,
    }
    coreg, defo, fit, pars = mdreg.fit(
        data['array'],                  # Signal data to correct
        fit_image = vfa_fit,            # Signal model
        fit_coreg = coreg_params,       # Coregistration model
        maxit = 5,                      # Maximum number of iteration
        verbose = 2,
    )
    plot_settings = {
        'interval' : 500,                   # Time between animation frames in ms
        'vmin' : 0,                         # Minimum value of the colorbar
        'vmax' : np.percentile(data['array'],99),   # Maximum value of the colorbar
        'show' : True,                      # Display the animation on screen
    }
    anim = mdreg.animation(
        coreg, 
        title = 'Motion corrected', 
        **plot_settings,
    )


if __name__ == '__main__':

#    test_image_model()
#    test_pixel_model()
#    test_coreg_model()
    test_vfa_lin()