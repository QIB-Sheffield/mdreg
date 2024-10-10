"""
===============================================
Example 3D data: Variable Flip Angle
===============================================

This example illustrates coregistration of a 3D Variable Flip Angle (VFA) dataset. 
The data is fetched using the `fetch` function, and the desired slice is extracted from the data array. 
The VFA parameters are defined, and the coregistration parameters are set. 
The model-driven coregistration is performed, and the results are visualized.


"""

#%% 
# Import packages and data
# ----------------------------
# Example data can be easily loaded in using the `fetch` function.

import numpy as np
import mdreg
import time

data = mdreg.fetch('VFA')

#%% 
# Extract the desired slice from the data array
# ----------------------------
# As an intial step, we will extract the 4D data (x,y,z,t) from the fetched data dictionary.
array = data['array']

#%%
# Signal model theory
# ----------------------------
# The signal model used in this example is the non-linear variable flip angle SPGR model.
# The signal model is defined by the following equation:
#
#
# :math:`S(\phi)=S_{0} \frac{\sin{\phi}e^{-\frac{T_{R}}{T_{1}}}}{1-\cos{\phi}e^{-\frac{T_{R}}{T_{1}}}}`
#
# Where :math:`S` is the signal, :math:`S_{0}` the initial signal, :math:`\phi` the flip angle,
# :math:`T_{R}` the reptition time and :math:`T_{1}` the longitudinal relaxtion time.  Using this equation,
# :math:`T_{1}` and :math:`S_{0}` are optimised via a least squares method.
#

#%%
# Define model fit parameters
# ----------------------------
# The image fitting settings dictionary (`vfa_fit` in this case) is required by `mdreg.fit` to fit a specific signal model to the data.
# Leaving this as None will fit a constant model to the data as a default.
#
# Here, we select the model function `func` to be the non-linear varaible flip angle SPGR model from the model library (`mdreg.spgr_vfa_nonlin`).
# This model fit requires the flip angle values in radians (`FA`) and the repetition time in seconds (`TR`).
# This information is provided in the `data` dictionary for this example

vfa_fit = {
    'func': mdreg.spgr_vfa_nonlin,  # The function to fit the data
    'FA': np.deg2rad(np.array(data['FA'])),  # The FA values in radians
    'TR': 3.71/1000  # The TR value in seconds
}

#%%
# Define the coregistration parameters
# ----------------------------
# The coregistration parameters are set in the `coreg_params` dictionary.
# The `package` key specifies the coregistration package to be used, with a choice of elastix, skimage, or dipy.
# The `params` key specifies the parameters required by the chosen coregistration package. Here None is used to 
# specify default parameters for freeform registration included by `mdreg`.
# Here, we use the elastix package with the following parameters:

print(data['spacing'])

coreg_params = {
    'package': 'elastix',
    'params': None,
}

#%%
# Define the plotting parameters
# ----------------------------
# The plotting parameters are set in the `plot_settings` dictionary.
# The `interval` key specifies the time interval between frames in milliseconds, and the `vmin`/`vmax` keys specify the minimum/maximum value of the colorbar.
# The `slice` key specifies the slice to be displayed in the animation. If unset or set to None, the central slice is displayed by default.
# If you are interested to save the resulting animation, you can use the `path` key to the desired file path and the `filename` key to the desired filename.
# As a default these are set to None resulting in the animation being displayed on screen only.
# For more plotting keyword arguements, see the `mdreg.plot` module.
# 

plot_settings = {
    'interval' : 500,
    'vmin' : 0,
    'vmax' : np.percentile(array,99),
    'slice' : 10,
    'path' : None,
    'show' : True,
}

#%% 
# Perform model-driven coregistration
# ----------------------------
# The `mdreg.fit` function is used to perform the model-driven coregistration.
# The function requires the 4D data array, the fit_image dictionary, and the coregistration parameters we have already defined.
# The `verbose` parameter can be set to 0, 1, 2, or 3 to control the level of output.

non_lin_time = time.time()

coreg_nonlin, defo_nonlin, fit_nonlin, pars_nonlin = mdreg.fit(array,
                                                                fit_image=vfa_fit, 
                                                                fit_coreg=coreg_params, 
                                                                maxit=3, 
                                                                verbose=0)

print(f"Non linear fitting time elapsed: {(time.time() - non_lin_time):.3}s")

#%% 
# Visualise coregistration results
# ---------------------------------
# To easily visualise the output of the employ the `mdreg.plot` module to easily 
# produce animations that render on screen or save as a gif.
# Here we utilise `mdreg.plot_series` which accepts both 2D and 3D spatial data arrays 
# which change over an additional dimension (e.g. time or FA in this case). 
# This displays the orginal data, the fitted data and the coregistered data. For the case of 3D data,
# the function will display the central slice over the additional dimension
# unless an alternative slice is specified in the plotting parameters.
#

mdreg.plot_series(array, fit_nonlin, coreg_nonlin, **plot_settings)

#%% 
# Optimising the fitting process
# ------------------------------
# Applying this non-linear fitting in a pixel wise approach quickly becomes unfeasible with dynamic 3D
# series due to size of the datasets.
# 
# A linearised fitting approach can offer some efficiency gains. The above relationship can be rearranged to give:
#
# :math:`S\frac{\cos{\phi}}{\sin{\phi}}=M S\sin{\phi}+C`
#
# This takes the form: :math:`Y=MX+C`. Which can be easily solved for using linear methods.
# 
# In this case the slope and intercept terms are defined as:
#
# :math:`M=(\frac{1}{E});~~~~C=\frac{(1-E)}{E}~`. :math:`~~~~` Where :math:`E=e^{\frac{-T_{R}}{T_{1}}}~`.
#
# The fitted intercept and slope terms are then used to calculate the fitted signal :math:`S`.
#
# :math:`S(\phi)=\frac{C\sin{\phi}}{\cos{\phi}-M}`
#

#%%
# Compare with the linearised model
# ----------------------------
# A linearised model can also be used to fit the data. 
# Defined in the `mdreg.spgr_vfa_lin` function, this model also requires the flip angle values in radians (`FA`).
#
# The `mdreg` model fit parameters require deifnition as before, but the `func` key is now set to `mdreg.spgr_vfa_lin`.
#
# We utilise the same plotting parameters as before.

vfa_lin_fit = {
    'func': mdreg.spgr_vfa_lin,  # The function to fit the data
    'FA': np.deg2rad(np.array(data['FA'])),  # The FA values in radians
    }

lin_time = time.time()

coreg_lin, defo_lin, fit_lin, pars_lin = mdreg.fit(array, 
                                                   fit_image=vfa_lin_fit, 
                                                   fit_coreg=coreg_params,
                                                   maxit=1, 
                                                   verbose=0,
                                                   plot_params=plot_settings)

print(f"Linear fitting time elapsed: {(time.time() - lin_time):.3}s")

#%% 
# Visualize the results
# ----------------------------
# Again the `mdreg.plot_series` function can be used to visualize the results of the coregistration.
# Here are shown the results of the linearised model fits.

mdreg.plot_series(array, fit_lin, coreg_lin, **plot_settings)

# sphinx_gallery_start_ignore

# Choose the last image as a thumbnail for the gallery
# sphinx_gallery_thumbnail_number = -1

# sphinx_gallery_end_ignore