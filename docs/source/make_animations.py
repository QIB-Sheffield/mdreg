
import mdreg
import numpy as np

path="docs/source/_static/animations/"


def fetch_molli():

    # fetch the data
    data = mdreg.fetch('MOLLI')

    # We will consider the slice z=0 of the data array:
    array = data['array'][:,:,0,:]

    # Use the built-in animation function of mdreg to visualise the motion:
    mdreg.animation(array, path=path, filename='fetch_molli', vmin=0, vmax=1e4)



def mdreg_default():

    # fetch the data
    data = mdreg.fetch('MOLLI')

    # We will consider the slice z=0 of the data array:
    array = data['array'][:,:,0,:]

    # Perform model-driven coregistration
    coreg, defo, fit, pars = mdreg.fit(array)

    # Visualise the results
    mdreg.plot_series(array, fit, coreg, path=path, filename='mdreg_default', vmin=0, vmax=1e4)


def molli_builtin():

    # fetch the data
    data = mdreg.fetch('MOLLI')

    # We will consider the slice z=0 of the data array:
    array = data['array'][:,:,0,:]

    # Perform model-driven coregistration
    coreg, defo, fit, pars = mdreg.fit(array,
        fit_image = {
            'func': mdreg.abs_exp_recovery_2p,
            'TI': np.array(data['TI'])/1000,
            'parallel': True,
        },
        verbose=2,
    )

    # Visualise the results
    mdreg.plot_series(array, fit, coreg, path=path, filename='molli_builtin', vmin=0, vmax=1e4)



def my_pixel_model(xdata, S0, T1):
    return np.abs(S0 * (1 - 2 * np.exp(-xdata/T1)))

def my_pixel_model_init(xdata, ydata, p0):
    S0 = np.amax(np.abs(ydata))
    return [S0*p0[0], p0[1]]

def molli_my_fit():

    # fetch the data
    data = mdreg.fetch('MOLLI')

    # We will consider the slice z=0 of the data array:
    array = data['array'][:,:,0,:]

    my_pixel_fit = {
        'model': my_pixel_model,
        'xdata': np.array(data['TI'])/1000,
        'func_init': my_pixel_model_init,
        'bounds': (0, np.inf),
        'p0': [1,1.3],  
    }  

    # Coregister with the t1-model:
    coreg, defo, fit, pars = mdreg.fit(array,
        fit_pixel = my_pixel_fit,
        verbose=2,
    )

    # Show the result with mdreg's built-in plot functions
    mdreg.plot_series(array, fit, coreg, path=path, filename='molli_my_fit', vmin=0, vmax=1e4)





def molli_skimage():

    # fetch the data
    data = mdreg.fetch('MOLLI')

    # We will consider the slice z=0 of the data array:
    array = data['array'][:,:,0,:]

    # Perform model-driven coregistration
    coreg, defo, fit, pars = mdreg.fit(array,
        fit_image = {
            'func': mdreg.abs_exp_recovery_2p,
            'TI': np.array(data['TI'])/1000,
            'parallel': True,
        },
        fit_coreg = {
            'package': 'skimage',
            'params': {
                'attachment': 30.0,
            },
        },
        verbose=2,
    )

    # Visualise the results
    mdreg.plot_series(array, fit, coreg, path=path, filename='molli_skimage', vmin=0, vmax=1e4)



if __name__ == '__main__':

    # fetch_molli()
    mdreg_default()
    # molli_builtin()
    # molli_my_fit()
    # molli_custom_coreg()
    # molli_skimage()