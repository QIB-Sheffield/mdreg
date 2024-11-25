"""
===============================================
Example 3D data: Variable Flip Angle
===============================================

This example illustrates coregistration of a 3D Variable Flip Angle (VFA) 
dataset. The data is fetched using the `fetch` function, and the desired slice 
is extracted from the data array. The VFA parameters are defined, and the 
coregistration parameters are set. The model-driven coregistration is 
performed, and the results are visualized.

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
# As an intial step, we will extract the 4D data (x,y,z,t) from the fetched 
# data dictionary.

array = data['array']

#%%
# Signal model theory
# ----------------------------
# The signal model used in this example is the variable flip angle 
# SPGR model. The signal model is defined by the following equation:
#
#
# :math:`S(\phi)=S_{0} \frac{\sin{\phi}e^{-\frac{T_{R}}{T_{1}}}}{1-\cos{\phi}e^{-\frac{T_{R}}{T_{1}}}}`
#
# Where :math:`S` is the signal, :math:`S_{0}` the initial signal, :math:`\phi`
# the flip angle, :math:`T_{R}` the reptition time and :math:`T_{1}` the 
# longitudinal relaxtion time.
# 
# A linearised fitting approach is chosen for efficiency, where the above 
# relationship can be rearranged to give:
#
# :math:`S\frac{\cos{\phi}}{\sin{\phi}}=M S\sin{\phi}+C`
#
# This takes the form: :math:`Y=MX+C`. Which can be easily solved for using 
# linear methods.
# 
# In this case the slope and intercept terms are defined as:
#
# :math:`M=(\frac{1}{E});~~~~C=\frac{(1-E)}{E}~`. :math:`~~~~` Where 
# :math:`E=e^{\frac{-T_{R}}{T_{1}}}~`.
#
# The fitted intercept and slope terms are then used to calculate the fitted 
# signal :math:`S`.
#
# :math:`S(\phi)=\frac{C\sin{\phi}}{\cos{\phi}-M}`
#

#%%
# Define model fit parameters
# ----------------------------
# The image fitting settings dictionary (`vfa_fit` in this case) is 
# required by `mdreg.fit` to fit a specific signal model to the data. Leaving 
# this as None will fit a constant model to the data as a default.
#
# Here, we select the model function `func` to be the linear varaible flip 
# angle SPGR model from the model library (`mdreg.spgr_vfa_lin`). This model 
# fit requires the flip angle values in radians (`FA`). This information is 
# provided in the `data` dictionary for this example.

vfa_fit = {
    'func': mdreg.spgr_vfa_lin,  # The function to fit the data
    'FA': data['FA']  # The FA values in degrees
    }
#%%
# Define the coregistration parameters
# ----------------------------
# The coregistration parameters are set in the `coreg_params` dictionary.
# The `package` key specifies the coregistration package to be used, with a 
# choice of elastix, skimage, or dipy.
#
# The `params` key specifies the parameters required by the chosen 
# coregistration package. Here None is used to specify default parameters for 
# freeform registration included by `mdreg`. Here, we use the elastix package 
# with the following parameters:


coreg_params = {
    'package': 'elastix',
    'params': mdreg.elastix.params(FinalGridSpacingInPhysicalUnits='150.0'),
    'spacing': data['spacing']
}

#%%
# Define the plotting parameters
# ----------------------------
# The plotting parameters are set in the `plot_settings` dictionary.
#
# The `interval` key specifies the time interval between frames in 
# milliseconds, and the `vmin`/`vmax` keys specify the minimum/maximum value of 
# the colorbar. 
# 
# For the case of 3D data, by default the function renders animations for all 
# slices for the original, fitted and coregistered data in seperate figures. If
# the `slice` parameter is specified in the plotting parameters, the function
# will produce a single figure for the specified slice showing the original,
# fitted and coregistered data animations side-by-side.
# 
# If you are interested to save the resulting animation, you can use 
# the `path` key to the desired file path and the `filename` key to the desired 
# filename. As a default these are set to None resulting in the animation being 
# displayed on screen only. For more plotting keyword arguements, see the 
# `mdreg.plot` module.
# 
# The plotting parameters can be provide to the `mdreg.fit` function to provide
# settings for animations produce if verbose is set to 3. This produces
# animations after each coregistation iteration. Additionally, the plotting
# parameters can be provided to the `mdreg.plot` module to produce animations
# of the final outputs.

plot_settings = {
    'interval' : 500, # Time interval between animation frames in ms
    'vmin' : 0, # Minimum value of the colorbar
    'vmax' : np.percentile(array,99), # Maximum value of the colorbar
    'path' : None, # Path to save the animation
    'show' : True, # Display the animation on screen
    'filename' : None, # Filename to save the animation
    'slice' : None # No slice specified, show all slices for 3D data
}

#%% 
# Perform model-driven coregistration
# ----------------------------
# The `mdreg.fit` function is used to perform the model-driven coregistration.
# The function requires the 4D data array, the fit_image dictionary, and the 
# coregistration parameters we have already defined.
# The `verbose` parameter can be set to 0, 1, 2, or 3 to control the level of 
# output.

stime = time.time()

coreg, defo, fit, pars = mdreg.fit(array, 
                                   fit_image = vfa_fit, 
                                   fit_coreg = coreg_params,
                                   maxit = 3, 
                                   verbose = 0,
                                   plot_params = None)

tot_time = time.time() - stime

print(f"Linear fitting time elapsed: {(int(tot_time/60))} mins, {np.round(tot_time-(int(tot_time/60)*60),1)} s")

#%% 
# Visualize the results
# ----------------------------
# To easily visualise the output of the employ the `mdreg.plot` module to 
# easily produce animations that render on screen or save as a gif.
# Here we utilise `mdreg.plot_series` which accepts both 2D and 3D spatial data 
# arrays which change over an additional dimension (e.g. time or FA in this 
# case). This displays the orginal data, the fitted data and the coregistered 
# data. 
# 
# Here we apply the plotting parameters defined above to visualise the final
# results.

anim = mdreg.animation(array, title='Original Data', **plot_settings)

#%%
anim = mdreg.animation(coreg, title='Coregistered', **plot_settings)

#%%
anim = mdreg.animation(fit, title='Model Fit', **plot_settings)

#%% 
# Export all series at once
# ----------------------------
# The `mdreg.plot_series` function can be used to plot the original, fitted and
# coregistered data for all slices in the data array. This function can also
# be used to save the animations to a file. 
#
# We include the `mdreg.animation` function to display the animations on screen
# within the documentation, but recommend using the `mdreg.plot_series` 
# function to easily process and save the animations to a file when running 
# locally.
#  >>> anims = mdreg.plot_series(array, fit_nonlin, coreg_nonlin, **plot_settings)

# sphinx_gallery_start_ignore

# Choose the last image as a thumbnail for the gallery
# sphinx_gallery_thumbnail_number = -1

# sphinx_gallery_end_ignore