"""
===============================================
Using custom-built models
===============================================

In case an appropriate signal model is not available in mdreg's model 
library, the model must be custom writtem. This tutorial will illustrate this 
for the case of T1-mapping with a MOLLI sequence.

"""

#%%
# Import packages and data
# ----------------------------
# Import packages 
import numpy as np
from scipy.optimize import curve_fit
import mdreg

#%%
# Load test data
data = mdreg.fetch('MOLLI')
array = data['array'][:,:,0,:]

# Show the motion:
mdreg.animation(array, vmin=0, vmax=1e4, show=True)

#%%
# Building a custom model
# -----------------------
#
# The appropriate model for a single pixel signal has two free parameters 
# S0 and T1, apart from the inversion time TI, which is a sequence parameter:

def my_molli(TI, S0, T1):
    return np.abs(S0 * (1 - 2 * np.exp(-TI/T1)))

#%%
# In order to use this as a model fit in mdreg, we need a function that 
# takes the image array as argument, and returns the fit to the model and the 
# fitted model parameters:

def fit_my_molli(array, TI=None):

    # Reshape the image array for convenience
    shape = array.shape
    array = array.reshape((-1,shape[-1]))

    # Define output arrays in the new shape
    nx, nt = array.shape
    par = np.zeros((nx, 2))
    fit = np.zeros((nx, nt))

    # Fit the model for each pixel
    for x in range(nx):
        try:
            par[x,:], _ = curve_fit(
                my_molli, TI, array[x,:], 
                p0 = [np.amax(np.abs(array[x,:])), 1.3], 
                bounds = (0, np.inf),
            )
            fit[x,:] = my_molli(TI, par[x,0], par[x,1])
        except:
            fit[x,:] = array[x,:]

    # Return results in the original shape
    return fit.reshape(shape), par.reshape(shape[:-1]+(2,))

#%%
# We can now use this custom model in the same way as built-in models when we 
# run ``mdreg``:

# Define the fit function and its arguments
my_fit = {
    'func': fit_my_molli,
    'TI': np.array(data['TI'])/1000,
}

# Perform model-driven coregistration
coreg, defo, fit, pars = mdreg.fit(array, fit_image=my_fit)

# Visualise the results
mdreg.plot_series(array, fit, coreg, vmin=0, vmax=1e4, show=True)

#%%
# In this case, the same result can be obtained using the built-in function  
# `~mdreg.abs_exp_recovery_2p`.


#%%
# Pixel-by-pixel fitting
# ----------------------
# The example above represents a common scenario where a 1D 
# signal model is applied for each pixel independently (*pixel-by-pixel 
# fitting*). For such models, ``mdreg`` offers a convenient shortcut which 
# removes the need to define a fit function explicitly. 
# 
# We illustrate this idea by fitting the *my_molli()* function 
# defined above. In this case we also need a function that derives initial 
# values from the data and any constant initial values p0 provided by the user:

def my_molli_init(TI, ydata, p0):
    S0 = np.amax(np.abs(ydata))
    return [S0*p0[0], p0[1]]

#%%
# With these definitions in hand, a pixel model fit can be defined as a 
# dictionary specifying the model, its parameters (xdata), and any optional 
# arguments:

my_pixel_fit = {

    # The custom-built single pixel model
    'model': my_molli,

    # xdata for the single-pixel model
    'xdata': np.array(data['TI'])/1000,

    # Optional: custom-built initialization function
    'func_init': my_molli_init,

    # Optional: initial values for the free parameters
    'p0': [1,1.3], 

    # Optional: bounds for the free model parameters
    'bounds': (0, np.inf),

    # Optional: any keyword arguments accepted by scipy's curve_fit
    'xtol': 1e-3,
}   

#%%
# And this can be provided directly to `~mdreg.mdreg.fit` via the keyword argument 
# *fit_pixel** - instructing ``mdreg`` to perform pixel-based fitting using 
# the parameters defined in *my_pixel_fit*:

# Perform model-driven coregistration with a custom pixel model
coreg, defo, fit, pars = mdreg.fit(array, fit_pixel=my_pixel_fit)

# Visualise the results
mdreg.plot_series(array, fit, coreg, vmin=0, vmax=1e4, show=True)

#%%
#
# As expected, the result is the same as before using the built-in model 
# `~mdreg.abs_exp_recovery_2p` and the custom-built function *fit_my_molli*.

# sphinx_gallery_start_ignore

# Choose the last image as a thumbnail for the gallery
# sphinx_gallery_thumbnail_number = -1

# sphinx_gallery_end_ignore