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

    params = mdreg.elastix.params('freeform')
    params["FinalGridSpacingInPhysicalUnits"] = "5.0"

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
            'params': params,
            'spacing': data['pixel_spacing'],
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

    params = mdreg.elastix.params('freeform')
    params["FinalGridSpacingInPhysicalUnits"] = "5.0"

    coreg, defo, fit, pars = mdreg.fit(moving,
        fit_image = {
            'func': mdreg.abs_exp_recovery_2p,
            'TI': np.array(data['TI'])/1000,
        },
        fit_coreg = {
            'spacing': data['pixel_spacing'],
            'params': params,
        },
        maxit=2, verbose=3,
        plot_params = {'path':path, 'vmin':0, 'vmax':vmax},
    )

def test_coreg_model():

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
            'package': 'skimage',
            'params': mdreg.skimage.params(attachment=30)
        },
        maxit=5, verbose=3,
        plot_params = {'path':path, 'vmin':0, 'vmax':vmax},
    )



if __name__ == '__main__':

#    test_image_model()
#    test_pixel_model()
    test_coreg_model()